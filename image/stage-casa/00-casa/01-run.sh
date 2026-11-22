#!/bin/bash -e
# CaSa image: no CaSa code inside – the setup wizard downloads the newest CaSa release from GitHub at the
# first login, so the image never needs to be rebuilt for a new CaSa version.
# Prepared here: a Python environment with the packages of requirements.txt (fast first start), the setup
# wizard, the update command and the start of the wizard at the first login.

install -d "${ROOTFS_DIR}/opt/casa-venv-setup"
install -m 644 files/requirements.txt "${ROOTFS_DIR}/opt/casa-venv-setup/requirements.txt"
on_chroot << 'EOF'
python3 -m venv /opt/casa-venv
/opt/casa-venv/bin/python -m pip install --no-cache-dir --upgrade pip
/opt/casa-venv/bin/python -m pip install --no-cache-dir -r /opt/casa-venv-setup/requirements.txt
rm -rf /opt/casa-venv-setup
EOF

install -d "${ROOTFS_DIR}/usr/local/lib/casa"
install -m 755 files/casa-setup "${ROOTFS_DIR}/usr/local/bin/casa-setup"
install -m 755 files/casa-update "${ROOTFS_DIR}/usr/local/bin/casa-update"
install -m 644 files/casa_config.py "${ROOTFS_DIR}/usr/local/lib/casa/casa_config.py"
install -m 644 files/casa-setup.sh "${ROOTFS_DIR}/etc/profile.d/casa-setup.sh"
install -m 644 files/casa-image "${ROOTFS_DIR}/etc/casa-image"
install -m 644 files/motd "${ROOTFS_DIR}/etc/motd"
