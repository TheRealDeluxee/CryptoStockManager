# CaSa image: the setup wizard starts at the login until it has been completed once.
# Later again (keyboard, drive password, keys): casa-setup
if [ -f /etc/casa-image ] && [ ! -f /etc/casa-setup-done ] && [ -t 0 ] && [ -t 1 ] && [ "$(id -u)" != 0 ]; then
    casa-setup
fi
