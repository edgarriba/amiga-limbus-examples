from abc import ABC, abstractmethod
from limbus.core import Component, Pipeline, Params, ComponentState 

from farm_ng.oak.client import OakCameraClient, OakCameraClientConfig

import numpy as np
import cv2
import kornia as K

import asyncio

class AmigaCamera(Component):

    def __init__(self, name: str):
        super().__init__(name)
        # configure the camera client
        self.config = OakCameraClientConfig(address="192.168.1.93", port=50051)
        self.client = OakCameraClient(self.config)

        self.stream = None

    @staticmethod
    def register_outputs():
        outputs = Params()
        outputs.declare("img")
        return outputs
  
    async def forward(self):
        if self.stream is None:
            await self.client.start_service()
            self.stream = self.client.stream_frames(every_n=10)

        response = await self.stream.read()
        frame = response.frame
        
        data: bytes = getattr(frame, "rgb").image_data

        # use imdecode function
        image = np.frombuffer(data, dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        self.outputs.set_param("img", image)
        return ComponentState.OK


class OpencvWindow(Component):

    @staticmethod
    def register_inputs():
        inputs = Params()
        inputs.declare("img")
        return inputs

    async def forward(self):
        img = self.inputs.get_param("img")
        cv2.imshow(f"{self.name}", img)
        cv2.waitKey(1)
        return ComponentState.OK


class KorniaProcess(Component):

    @staticmethod
    def register_inputs():
        inputs = Params()
        inputs.declare("img")
        return inputs

    @staticmethod
    def register_outputs():
        inputs = Params()
        inputs.declare("img")
        return inputs

    async def forward(self):
        img = self.inputs.get_param("img")

        img_t = K.image_to_tensor(img)
        img_t = img_t[None].float() / 255.
        img_t = K.filters.sobel(img_t, normalized=False)

        img = K.tensor_to_image(img_t)
        self.outputs.set_param("img", img)

        return ComponentState.OK


# NOTE: thos is the old scriptic mode
async def main():
    cam = AmigaCamera("oak1")
    viz1 = OpencvWindow("viz_raw")
    viz2 = OpencvWindow("viz_img")
    imgproc = KorniaProcess("imgproc")

    cam.outputs.img.connect(viz1.inputs.img)
    cam.outputs.img.connect(imgproc.inputs.img)
    imgproc.outputs.img.connect(viz2.inputs.img)

    pipeline = Pipeline()
    # NOTE: in future not needed
    pipeline.add_nodes([cam, viz1, viz2, imgproc])
    
    # run your pipeline
    # NOTE: in future not needed
    pipeline.traverse()
    await pipeline.async_execute()


class BaseApp(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline = Pipeline()
    
    def register_component(self, obj):
        self.pipeline.add_nodes(obj)
        setattr(self, obj.name, obj)
    
    @abstractmethod
    def register_components(self):
        raise NotImplementedError
    
    @abstractmethod
    def connect_components(self):
        raise NotImplementedError

    async def run(self):
        self.register_components()
        self.connect_components()
        self.pipeline.traverse()
        return await self.pipeline.async_execute()


class CameraApp(BaseApp):

    def register_components(self):
        self.register_component(AmigaCamera("cam"))
        self.register_component(OpencvWindow("viz1"))
        self.register_component(OpencvWindow("viz2"))
        self.register_component(KorniaProcess("imgproc"))
    
    def connect_components(self):
        self.cam.outputs.img >> self.viz1.inputs.img
        self.cam.outputs.img >> self.imgproc.inputs.img
        self.imgproc.outputs.img >> self.viz2.inputs.img


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(CameraApp().run())
    except asyncio.CancelledError as ex:
        pass
    loop.close()