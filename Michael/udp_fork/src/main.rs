extern crate futures;
#[macro_use]
extern crate tokio_core;

use std::error::Error;
use std::env;
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use futures::{Async, Future, Poll};
use tokio_core::net::UdpSocket;
use tokio_core::reactor;

fn main() {
    if let Err(e) = run() {
        println!("{}", e);
    }
}

fn run() -> Result<(), Box<Error>> {
    let local_ip = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
    let mut listener_port = 9000;

    let mut event_loop: reactor::Core = reactor::Core::new()?;

    // The base address is the eventual target for all UDP packets
    let base_port: u16 = env::args().nth(1).ok_or("Must provide source port")?.parse()?;
    let base_addr = SocketAddr::new(local_ip, base_port);

    let mut clients = vec![];
    for port in env::args().skip(2) {
        let port: u16 = port.parse()?;

        // For each port we want to forward data to we need to set up a corresponding listener port
        // for the client to send data back
        let listener_addr = SocketAddr::new(local_ip, listener_port);
        let listener_socket = UdpSocket::bind(&listener_addr, &event_loop.handle())?;
        listener_port += 1;

        clients.push(Client::new(SocketAddr::new(local_ip, port), listener_socket));
    }

    if clients.is_empty() {
        return Err("Must specific at least 1 client".into());
    }

    let socket = UdpSocket::bind(&base_addr, &event_loop.handle())?;
    let transfer = Transfer::new(socket, clients);

    event_loop.run(transfer)?;
    Ok(())
}

const BUFFER_SIZE: usize = 2 * 1024;

/// A future for managing the transfer of data from one socket to multiple other sockets.
struct Transfer {
    /// The base socket to foward data to and from
    socket: UdpSocket,

    /// The address to send data to the base
    dest_addr: Option<SocketAddr>,

    /// The clients to foward data to and receive data from
    clients: Vec<Client>,

    /// A buffer containing bytes read from the server that have yet to be written to all clients
    buffer: [u8; BUFFER_SIZE],

    /// The number of valid bytes in the buffer
    bytes_read: usize,
}

impl Transfer {
    fn new(socket: UdpSocket, clients: Vec<Client>) -> Transfer {
        Transfer {
            socket: socket,
            dest_addr: None,
            clients: clients,
            buffer: [0; BUFFER_SIZE],
            bytes_read: 0,
        }
    }

    /// Keeps track of sending data from the base socket to all clients
    fn base_to_clients(&mut self) -> Poll<(), io::Error> {
        // Write any remaining data to the clients
        for client in &mut self.clients {
            try_nb!(client.send(&self.buffer[..self.bytes_read]));
        }

        // Read some more data from the connected socket
        let (bytes_read, dest_addr) = try_nb!(self.socket.recv_from(&mut self.buffer));
        self.bytes_read = bytes_read;
        self.dest_addr = Some(dest_addr);

        // Prepare the clients for the next set of data
        for client in &mut self.clients {
            client.reset_write();
        }

        Ok(().into())
    }

    /// Keeps track of sending data from clients to the base socket
    fn client_to_base(&mut self) -> Poll<(), io::Error> {
        // We are unable to write data from the client until we have a destination for our data
        if let Some(ref addr) = self.dest_addr {

            // We need to be careful here to ensure that we do not end up mixing UDP streams
            // together.
            //
            // This requires two steps:
            //
            //  1. We attempt to fully write any pending data from any client. If the client is
            //     is unable to fully write the buffered data (due to the server socket returning
            //     Async::NotReady), this function immediately returns to prevent clients from
            //     sending interleaved buffers.
            //
            //  2. We read from all the clients filling their internal buffers.
            //
            // This is not 100% reliable, as the buffer may not contain the complete message,
            // however this is not possible to solve without understanding the encoding on the
            // message -- which is left up to the application.

            for client in &mut self.clients {
                try_nb!(client.write_to(&self.socket, addr));
            }

            let mut is_ready = false;
            for client in &mut self.clients {
                match client.recv() {
                    Ok(Async::NotReady) => {},
                    Ok(_) => is_ready = true,
                    Err(e) => return Err(e)
                }
            }

            // If any of the clients are ready to be read from, then this future is set to the ready
            // state.
            if is_ready { Ok(().into()) } else { Ok(Async::NotReady) }
        }
        else {
            Ok(Async::NotReady)
        }
    }
}

impl Future for Transfer {
    type Item = ();
    type Error = io::Error;

    fn poll(&mut self) -> Poll<(), io::Error> {
        loop {
            match (self.base_to_clients(), self.client_to_base()) {
                (Err(e), _) | (_, Err(e)) => return Err(e),
                (Ok(Async::NotReady), Ok(Async::NotReady)) => return Ok(Async::NotReady),
                _ => {}
            }
        }
    }
}

/// A Udp client that keeps track of how many bytes are written to it.
struct Client {
    /// The clients address
    addr: SocketAddr,

    /// The number of bytes sent to this client since its last reset
    bytes_written_self: usize,

    /// The listener socket for this client
    socket: UdpSocket,

    /// A buffer containing bytes read from the client
    buffer: [u8; BUFFER_SIZE],

    /// The number of valid bytes in the buffer
    bytes_read: usize,

    /// The number of bytes written to a destination address (out of the current )
    bytes_written_dest: usize,
}

impl Client {
    /// Create a new client from a target address
    fn new(addr: SocketAddr, socket: UdpSocket) -> Client {
        Client {
            addr: addr,
            socket: socket,
            bytes_written_self: 0,

            buffer: [0; BUFFER_SIZE],
            bytes_read: 0,
            bytes_written_dest: 0,
        }
    }

    /// Send data to the client
    fn send(&mut self, data: &[u8]) -> Poll<(), io::Error> {
        if self.bytes_written_self >= data.len() {
            return Ok(().into());
        }

        let remaining_data = &data[self.bytes_written_self..];
        self.bytes_written_self += try_nb!(self.socket.send_to(remaining_data, &self.addr));

        match self.bytes_written_self < data.len() {
            true => Ok(Async::NotReady),
            false => Ok(().into()),
        }
    }

    /// Reset the bytes written count for this client
    fn reset_write(&mut self) {
        self.bytes_written_self = 0;
    }

    /// Receive data from the client and store it in a buffer
    fn recv(&mut self) -> Poll<(), io::Error> {
        if self.bytes_written_dest < self.bytes_read {
            return Ok(Async::NotReady);
        }

        self.bytes_read = try_nb!(self.socket.recv_from(&mut self.buffer)).0;
        self.bytes_written_dest = 0;
        Ok(().into())
    }

    /// Write buffered data to a destination socket address
    fn write_to(&mut self, from_socket: &UdpSocket, dest: &SocketAddr) -> Poll<(), io::Error> {
        if self.bytes_written_dest >= self.bytes_read {
            return Ok(().into());
        }

        let remaining_data = &self.buffer[self.bytes_written_dest..self.bytes_read];
        self.bytes_written_dest += try_nb!(from_socket.send_to(remaining_data, dest));

         match self.bytes_written_self < self.bytes_read {
            true => Ok(Async::NotReady),
            false => Ok(().into()),
        }
    }
}
