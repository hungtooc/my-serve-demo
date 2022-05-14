export container=1c038dedeb36
export model_name=vgg16
echo '[1] delete existing model'
curl -X DELETE http://localhost:8081/models/vgg16
# echo '[2] remove old mar'
# rm vgg16.mar
echo '[3] remove mar in container'
docker exec -u root $container rm /home/model-server/model-store/vgg16.mar
echo '[4] build model.mar'
torch-model-archiver -f --model-name vgg16 --version 1.0 --model-file model.py --serialized-file vgg16-397923af.pth --handler vgg_handler.py --extra-files index_to_name.json,image_utils.py
echo '[5] copy mar to container'
docker cp vgg16.mar $container:/home/model-server/model-store
echo '[6] register model'
curl -X POST  "http://localhost:8081/models?url=$model_name.mar"
echo '[7] scaling worker'
curl -v -X PUT "http://localhost:8081/models/$model_name?min_worker=1"
echo '[8] predict request'
# curl http://127.0.0.1:8080/predictions/vgg16 -T /home/ubuntu/Projects/libs/serve/examples/image_classifier/kitten.jpg