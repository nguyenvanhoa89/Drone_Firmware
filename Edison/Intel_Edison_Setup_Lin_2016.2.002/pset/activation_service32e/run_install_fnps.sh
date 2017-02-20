#!/bin/sh

set +x

( cd "`dirname $0`/INTEL-11.10.1" && ./install_fnp.sh FNPLicensingService ) || exit 1
( cd "`dirname $0`/INTEL-11.11.0" && ./install_fnp.sh FNPLicensingService ) || exit 2
( cd "`dirname $0`/INTEL-11.11.1" && ./install_fnp.sh FNPLicensingService ) || exit 3
( cd "`dirname $0`/INTEL-11.12.0" && ./install_fnp.sh FNPLicensingService ) || exit 4
( cd "`dirname $0`/INTEL-11.13.0.2" && ./install_fnp.sh FNPLicensingService ) || exit 5

