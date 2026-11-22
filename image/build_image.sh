#!/bin/bash
# Builds the CaSa image for the Raspberry Pi Imager: Raspberry Pi OS Lite 64-bit (Debian 13 Trixie, without
# desktop) with the Python packages of CaSa, Samba, tmux and the setup wizard. No Docker in the image.
# CaSa itself is not in the image: the wizard downloads the newest release from GitHub at the first login,
# so the image only needs to be rebuilt for changes of the system, not for new CaSa versions.
# The build tool pi-gen runs in Docker Desktop (arm64 emulation). Takes about 1–2 hours.
#
#   bash image/build_image.sh            (in the CaSa folder; Windows: Git Bash)
#
# Result: image/deploy/<date>-casa.img.xz  ->  Raspberry Pi Imager: Choose OS > Use custom
# No CaSa files and never your private files (config.ini, transactions.csv, ...) go into the image.
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
FILES=image/stage-casa/00-casa/files
VERSION=$(sed -n 's/^VERSION = "\(.*\)".*/\1/p' main.py)
export MSYS_NO_PATHCONV=1   # Git Bash on Windows: do not change the paths inside the container
WIN_ROOT=$( (command -v cygpath >/dev/null && cygpath -w "$ROOT") || echo "$ROOT")

echo "=== CaSa image $VERSION"
echo "=== 1/3 Python packages list (CaSa itself comes from GitHub at the first login)"
cp requirements.txt "$FILES/requirements.txt"
echo "CaSa image, built $(date +%d/%m/%Y) (packages of CaSa $VERSION)" > "$FILES/casa-image"

echo "=== 2/3 Raspberry Pi OS + CaSa with pi-gen (about 1-2 hours)"
docker volume create casa-pigen >/dev/null
docker run --rm --privileged -v casa-pigen:/work -v "$WIN_ROOT/image:/casa-src:ro" debian:trixie bash -c '
set -e
apt-get update -qq
apt-get install -y -qq git ca-certificates >/dev/null
cd /work
[ -d pi-gen ] || git clone --depth 1 -b arm64 https://github.com/RPi-Distro/pi-gen.git
cd pi-gen
# depends: one program per line, as "program" or "program:package"
apt-get install -y -qq --no-install-recommends $(sed "s/^[^:]*://" depends) >/dev/null
# arm64 emulation of Docker Desktop (binfmt_misc) also visible in this container
grep -q binfmt_misc /proc/mounts || mount -t binfmt_misc binfmt_misc /proc/sys/fs/binfmt_misc
# our stage and settings, with Linux line endings and executable scripts
rm -rf /casa-image && cp -r /casa-src /casa-image
find /casa-image -type f ! -name "*.png" -exec sed -i "s/\r$//" {} +
chmod +x /casa-image/stage-casa/prerun.sh /casa-image/stage-casa/00-casa/01-run.sh
touch stage2/SKIP_IMAGES          # only the CaSa image, not the plain Lite image
rm -rf deploy work   # always a clean build (an interrupted build leaves a half-installed system)
./build.sh -c /casa-image/config
ls -la deploy
'

echo "=== 3/3 Result"
mkdir -p image/deploy
docker run --rm -v casa-pigen:/work -v "$WIN_ROOT/image/deploy:/out" debian:trixie sh -c 'cp /work/pi-gen/deploy/*.img.xz /out/ && ls -la /out'
echo "Done: image/deploy/*.img.xz  ->  Raspberry Pi Imager: Choose OS > Use custom"
echo "Clean up afterwards (frees about 10 GB): docker volume rm casa-pigen"
