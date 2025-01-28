nano %UserProfile%/.wslconfig
wsl --shutdown
docker restart $(docker ps -q)