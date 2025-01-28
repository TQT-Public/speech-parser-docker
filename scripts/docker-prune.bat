docker container prune
docker builder prune
docker image prune --all
docker volume prune --all
docker system prune --all --force --volumes
docker system df
docker restart $(docker ps -q)