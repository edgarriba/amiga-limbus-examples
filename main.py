import argparse
import asyncio

from limbus.core import Component, InputParams, OutputParams, ComponentState 
from limbus.core.app import App

from farm_ng.core.events_file_writer import EventsFileWriter
from farm_ng.oak.camera_client import OakCameraClient
from farm_ng.service.service_client import ClientConfig

import numpy as np
import cv2
import kornia as K
from turbojpeg import TurboJPEG


class AmigaCamera(Component):

    def __init__(self, name: str, address: str, port: int):
        super().__init__(name)
        # configure the camera client
        self.config = ClientConfig(address=address, port=port)
        self.client = OakCameraClient(self.config)

        # create a stream
        self.stream = self.client.stream_frames(every_n=1)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        outputs.declare("rgb")
        outputs.declare("disparity")
  
    async def forward(self):
        response = await self.stream.read()
        frame = response.frame
        
        await asyncio.gather(
            self._outputs.rgb.send(frame.rgb.image_data),
            self._outputs.disparity.send(frame.disparity))

        return ComponentState.OK


class ImageDecoder(Component):
    def __init__(self, name: str):
        super().__init__(name)
        self.decoder = TurboJPEG()
    
    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        inputs.declare("data")

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        outputs.declare("img")

    async def forward(self):
        data: bytes = await self._inputs.data.receive()
        image: np.ndarray = self.decoder.decode(data)
        await self._outputs.img.send(image)
        return ComponentState.OK


class OpencvWindow(Component):

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        inputs.declare("img")

    async def forward(self):
        img: np.ndarray = await self._inputs.img.receive()
        cv2.imshow(f"{self.name}", img)
        cv2.waitKey(1)
        return ComponentState.OK


class KorniaProcess(Component):

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        inputs.declare("img")

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        outputs.declare("img")

    async def forward(self):
        img = await self._inputs.img.receive()

        img_t = K.image_to_tensor(img)
        img_t = img_t[None].float() / 255.
        img_t = K.filters.sobel(img_t.cuda(), normalized=False)

        img = K.tensor_to_image(img_t)
        await self._outputs.img.send(img)

        return ComponentState.OK


class EventsWriter(Component):

    def __init__(self, name: str):
        super().__init__(name)
        self.writer = EventsFileWriter("amiga-limbus-app.bin")
        assert self.writer.open()

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        inputs.declare("data")

    async def forward(self):
        data: bytes = await self._inputs.data.receive()
        self.writer.write("disparity", data)
        return ComponentState.OK


class AmigaApp(App):

    def __init__(self, address: str, port: int) -> None:
        self.address = address
        self.port = port
        super().__init__()

    def create_components(self):
        self.cam = AmigaCamera("cam", address=self.address, port=self.port)
        self.decoder = ImageDecoder("decoder")
        self.viz1 = OpencvWindow("viz1")
        self.viz2 = OpencvWindow("viz2")
        self.imgproc = KorniaProcess("imgproc")
        self.writer = EventsWriter("writer")
    
    def connect_components(self):
        self.cam.outputs.rgb >> self.decoder.inputs.data
        self.cam.outputs.disparity >> self.writer.inputs.data
        self.decoder.outputs.img >> self.viz1.inputs.img
        self.decoder.outputs.img >> self.imgproc.inputs.img
        self.imgproc.outputs.img >> self.viz2.inputs.img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="amiga-limbus-app")
    parser.add_argument(
        "--address",
        type=str,
        required=True,
        help="The IP address of the Amiga camera.",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="The port of the Amiga camera.",
    )
    args = parser.parse_args()

    loop = asyncio.get_event_loop()

    app = AmigaApp(args.address, args.port)
    try:
        loop.run_until_complete(app.run())
    except asyncio.CancelledError as ex:
        pass
    loop.close()