#!/bin/bash -e
# CaSa stage: starts from the Raspberry Pi OS Lite root filesystem of stage2
if [ ! -d "${ROOTFS_DIR}" ]; then
	copy_previous
fi
