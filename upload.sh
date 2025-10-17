docker buildx create --name mybuilder --use
docker buildx inspect --bootstrap
docker login -u adityagoyal333
docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile -t adityagoyal333/rainyday:rainyday_img --push .

