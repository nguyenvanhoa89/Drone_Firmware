#! /bin/sh

# Client installer script
# =======================

trap 'echo ; exit 13' TERM INT

# Make sure run by sudo
USERID=`id | cut -d \( -f 1 | cut -d \= -f 2`
CHFLAGS=`which chflags 2> /dev/null`
CHATTR=`which chattr 2> /dev/null`
if [ "$SUDO_USER" ]
then
        OWNER=$SUDO_USER
else
        OWNER=$USER
fi

if [ "$USERID" -ne 0 ]
then
echo "Please run this script using 'sudo' or as root (su)."
    exit 1
fi


# If no $1 provided try ./bin/FNPLicensingService/FNPLicensingService and 
# ./FNPLicensingService/FNPLicensingService

if [ -z $1 ]
then
   if [ -f ./bin/FNPLicensingService/FNPLicensingService ]
   then
       service=./bin/FNPLicensingService/FNPLicensingService
   else
       service=./FNPLicensingService/FNPLicensingService
   fi
else
    service=$1
fi

# Check that service is available
if [ ! -f "$service" ]
then
    echo "Unable to locate anchor service to install, please specify correctly on command line"
    exit 1
fi

case `uname` in
    "Darwin")
        SERVICE_DEST="/Library/Application Support/FLEXnet Publisher/Service/11.10.1"
        ;;
    "Linux")
        if file ${service} | grep "ELF 64" > /dev/null
        then
            SERVICE_DEST="/usr/local/share/FNP/service64/11.10.1"
        else
            SERVICE_DEST="/usr/local/share/FNP/service/11.10.1"
        fi
        ;;
esac

echo "Installing anchor service from $service to $SERVICE_DEST"

# Copy FnpLicensingService

mkdir -p "${SERVICE_DEST}"

cp "${service}" "${SERVICE_DEST}"
chown root "${SERVICE_DEST}/FNPLicensingService"
chmod 4755 "${SERVICE_DEST}/FNPLicensingService"


echo 
echo "Checking system for trusted storage area..."

if [ -d '/Library/Preferences' ]
then
    platform="Mac OS X"
    rootpath="/Library/Preferences/FLEXnet Publisher"
else
    platform=`uname`
    rootpath="/usr/local/share/macrovision/storage"
fi

echo "Configuring for $platform, Trusted Storage path $rootpath..."

set +e

#
# Check for existance of directory, if not present try to create
#

if [ -d "$rootpath" ]
then
    if [ -x "$CHFLAGS" ]
    then
	find "$rootpath" -flags uchg -exec chflags nouchg "$rootpath/FLEXnet" {} \;
    elif [ -x "$CHATTR" ]
    then
        chattr -i "$rootpath/FLEXnet"
    fi
    chown $OWNER "$rootpath/FLEXnet"
    echo "$rootpath already exists..."
else
    echo "Creating $rootpath..."
    mkdir -p "$rootpath"
fi

#
# Set correct permissions on directory
#

echo "Setting permissions on $rootpath..."
chmod 777 "$rootpath"
rc=$?
if [ $rc -ne 0 ]
then
    echo "Unable to set permissions on $rootpath, chmod returned error $rc."
    exit $rc
fi

echo "Permissions set..."

# All done.

echo "Configuration completed successfully."
echo 
