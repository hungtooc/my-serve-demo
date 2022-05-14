import cv2
import importlib
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier
from ts.torch_handler.vision_handler import VisionHandler
from ts.utils.util import list_classes_from_module
from ts.utils.util  import map_class_to_label
from captum.attr import IntegratedGradients
from ts.torch_handler.base_handler import BaseHandler

from ts.utils.util import PredictionException
from image_utils import numpy_to_base64_String

from abc import ABC
from PIL import Image
import base64
import io
import time
from pkg_resources import packaging

if packaging.version.parse(torch.__version__) >= packaging.version.parse("1.8.1"):
    from torch.profiler import profile, record_function, ProfilerActivity
    PROFILER_AVAILABLE = True
else:
    PROFILER_AVAILABLE = False


class VGGImageClassifier(BaseHandler, ABC):
    """
    Overriding the model loading code as a workaround for issue :
    https://github.com/pytorch/serve/issues/535
    https://github.com/pytorch/vision/issues/2473
    """
    topk = 5
    # These are the standard Imagenet dimensions
    # and statistics
    image_processing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # @staticmethod
    # def numpy_to_base64_String(image):
    #     pil_image = Image.fromarray(image)
    #     rawBytes = io.BytesIO()
    #     pil_image.save(rawBytes, "PNG")
    #     rawBytes.seek(0)
    #     img_base64 = base64.b64encode(rawBytes.getvalue())
    #     return img_base64.decode()

    def set_max_result_classes(self, topk):
        self.topk = topk

    def get_max_result_classes(self):
        return self.topk

    def postprocess(self, data):
        ps = F.softmax(data, dim=1)
        probs, classes = torch.topk(ps, self.topk, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()
        log = {'read_image' : False}
        
        try:
            image = cv2.imread("/home/model-server/images/kim-anh.jpg")
            log["read_image"] = True
        except:
            raise PredictionException("cannot read image", 513)
            

        if log['read_image']:
            try:
                base64_image = numpy_to_base64_String(image) # VGGImageClassifier.
                
            except Exception as e:
                raise e

            log['base64_image'] = base64_image
        try:
            result = map_class_to_label(probs, self.mapping, classes)
        except:
            raise PredictionException("cannot predict classifer", 513)
        
        output = [{"pwd": str(os.getcwd()), "log": log, "result": result} for _ in range(len(probs))]
        try:
            return output
        except:
            raise PredictionException("cannot return output", 513)

        

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError("Expected only one class as model definition. {}".format(
                model_class_definitions))

        model_class = model_class_definitions[0]
        state_dict = torch.load(model_pt_path)
        model = model_class()
        model.load_state_dict(state_dict)
        return model

    def initialize(self, context):
        super().initialize(context)
        self.ig = IntegratedGradients(self.model)
        self.initialized = True
        properties = context.system_properties
        if not properties.get("limit_max_image_pixels"):
            Image.MAX_IMAGE_PIXELS = None

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    def get_insights(self, tensor_data, _, target=0):
        print("input shape", tensor_data.shape)
        return self.ig.attribute(tensor_data, target=target, n_steps=15).tolist()

    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.

        Returns:
            list : Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        is_profiler_enabled = os.environ.get("ENABLE_TORCH_PROFILER", None)
        if is_profiler_enabled:
            if PROFILER_AVAILABLE:
                output, _ = self._infer_with_profiler(data=data)
            else:
                raise RuntimeError("Profiler is enabled but current version of torch does not support."
                                   "Install torch>=1.8.1 to use profiler.")
        else:
            if self._is_describe():
                output = [self.describe_handle()]
            else:
                data_preprocess = self.preprocess(data)

                if not self._is_explain():
                    output = self.inference(data_preprocess)
                    output = self.postprocess(output)
                else:
                    output = self.explain_handle(data_preprocess, data)

        stop_time = time.time()
        metrics.add_time('HandlerTime', round(
            (stop_time - start_time) * 1000, 2), None, 'ms')
        return output