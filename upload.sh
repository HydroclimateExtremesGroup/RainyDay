docker buildx create --name mybuilder --use
docker buildx inspect --bootstrap
docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile -t rainyday_img:latest --push .
docker tag rainyday_img:latest adityagoyal333/Rainyday:rainyday_img
docker push adityagoyal333/Rainyday:rainyday_img

