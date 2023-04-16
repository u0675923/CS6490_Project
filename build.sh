#!/bin/sh

docker run --volume "$(pwd)":/home/user/hostcwd buildozer $@
