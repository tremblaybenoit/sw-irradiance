sudo lsblk #list disks that are attached
sudo mkdir -p /mnt/miniset #making directory 
sudo mount -o discard,defaults /dev/sdb /mnt/miniset 