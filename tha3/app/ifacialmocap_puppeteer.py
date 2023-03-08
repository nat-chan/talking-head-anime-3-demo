import argparse
import os
import socket
import sys
import threading
import time
from typing import Optional, Union
from collections import defaultdict

sys.path.append(os.getcwd())

from tha3.mocap.ifacialmocap_pose import create_default_ifacialmocap_pose
from tha3.mocap.ifacialmocap_v2 import IFACIALMOCAP_PORT, IFACIALMOCAP_START_STRING, parse_ifacialmocap_v2_pose
from tha3.poser.modes.load_poser import load_poser

import torch
import wx

from tha3.poser.poser import Poser
from tha3.mocap.ifacialmocap_constants import RIGHT_EYE_BONE_ROTATIONS, LEFT_EYE_BONE_ROTATIONS, HEAD_BONE_ROTATIONS, ROTATION_NAMES
from tha3.mocap.ifacialmocap_pose_converter import IFacialMocapPoseConverter
from tha3.util import torch_linear_to_srgb, resize_PIL_image, extract_PIL_image_from_filelike, \
    extract_pytorch_image_from_PIL_image


def convert_linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    rgb_image = torch_linear_to_srgb(image[0:3, :, :])
    return torch.cat([rgb_image, image[3:4, :, :]], dim=0)

import numpy as np
class FifoArray:
    def __init__(
        self,
        fps: int = 28,
        target_ms: int = 1015,
    ) -> None:
        self.fifo_length = int(fps*target_ms/1000)
        self.array = np.zeros((self.fifo_length, 512, 512, 4), dtype=np.uint8)
        self.i = 0

    def push(self, frame: np.ndarray) -> None:
        self.array[self.i] = frame
        self.i = (self.i + 1) % self.fifo_length

    def pop(self) -> np.ndarray:
        return self.array[self.i]

class MyStatistics:
    def __init__(self):
        self.count = 100
        self.defaultdict = defaultdict(list)

    def add(self, key: str, value: Union[float, int]) -> None:
        self.defaultdict[key].append(value)
        while len(self.defaultdict[key]) > self.count:
            del self.defaultdict[key][0]

    def get(self, key: str) -> Union[float, int]:
        if len(self.defaultdict[key]) == 0:
            return 0.0
        else:
            return sum(self.defaultdict[key])/len(self.defaultdict[key])
    def txt(self, key: str, value: Union[float, int]) -> str:
        self.add(key, value)
        return f"{key}={self.get(key)/10**6:.2f} "


class MainFrame(wx.Frame):
    def __init__(self, poser: Poser, pose_converter: IFacialMocapPoseConverter, device: torch.device):
        super().__init__(None, wx.ID_ANY, "CuTalk")
        self.pose_converter = pose_converter
        self.poser = poser
        self.device = device


        self.ifacialmocap_pose = create_default_ifacialmocap_pose()
        self.source_image_bitmap = wx.Bitmap(self.poser.get_image_size(), self.poser.get_image_size())
        self.result_image_bitmap = wx.Bitmap(self.poser.get_image_size(), self.poser.get_image_size())
        self.wx_source_image = None
        self.torch_source_image = None
        self.last_pose: Optional[list[float]] = None
        self.statistics = MyStatistics()
        self.last_update_time = None

        self.create_receiving_socket()
        self.create_ui()
        self.create_timers()
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.update_source_image_bitmap()
        self.update_result_image_bitmap()
        self.fifo_array = FifoArray(fps=28, target_ms=1015)
        self.load_image_direct(r"C:\Users\qzrp0\talking-head-anime-3-demo\data\images\yukarin.png")

    def create_receiving_socket(self):
        self.receiving_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiving_socket.bind(("", IFACIALMOCAP_PORT))
        self.receiving_socket.setblocking(False)

    def create_timers(self):
        self.capture_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_capture_panel, id=self.capture_timer.GetId())
        self.animation_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_result_image_bitmap, id=self.animation_timer.GetId())

    def on_close(self, event: wx.Event):
        # Stop the timers
        self.animation_timer.Stop()
        self.capture_timer.Stop()

        # Close receiving socket
        self.receiving_socket.close()

        # Destroy the windows
        self.Destroy()
        event.Skip()

    def on_start_capture(self, event: wx.Event):
        capture_device_ip_address = self.capture_device_ip_text_ctrl.GetValue()
        out_socket = None
        try:
            address = (capture_device_ip_address, IFACIALMOCAP_PORT)
            out_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            out_socket.sendto(IFACIALMOCAP_START_STRING, address)
        except Exception as e:
            message_dialog = wx.MessageDialog(self, str(e), "Error!", wx.OK)
            message_dialog.ShowModal()
            message_dialog.Destroy()
        finally:
            if out_socket is not None:
                out_socket.close()

    def read_ifacialmocap_pose(self):
        if not self.animation_timer.IsRunning():
            return self.ifacialmocap_pose
        socket_bytes = None
        while True:
            try:
                socket_bytes = self.receiving_socket.recv(8192)
            except socket.error as e:
                break
        if socket_bytes is not None:
            socket_string = socket_bytes.decode("utf-8")
            self.ifacialmocap_pose = parse_ifacialmocap_v2_pose(socket_string)
        return self.ifacialmocap_pose

    def on_erase_background(self, event: wx.Event):
        pass

    def create_animation_panel(self, parent):
        self.animation_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.animation_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.animation_panel.SetSizer(self.animation_panel_sizer)
        self.animation_panel.SetAutoLayout(1)

        image_size = self.poser.get_image_size()

        if True:
            self.input_panel = wx.Panel(self.animation_panel, size=(image_size, image_size + 128),
                                        style=wx.SIMPLE_BORDER)
            self.input_panel_sizer = wx.BoxSizer(wx.VERTICAL)
            self.input_panel.SetSizer(self.input_panel_sizer)
            self.input_panel.SetAutoLayout(1)
            self.animation_panel_sizer.Add(self.input_panel, 0, wx.FIXED_MINSIZE)

            self.source_image_panel = wx.Panel(self.input_panel, size=(image_size, image_size), style=wx.SIMPLE_BORDER)
            self.source_image_panel.Bind(wx.EVT_PAINT, self.paint_source_image_panel)
            self.source_image_panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
            self.input_panel_sizer.Add(self.source_image_panel, 0, wx.FIXED_MINSIZE)

            self.load_image_button = wx.Button(self.input_panel, wx.ID_ANY, "Load Image")
            self.input_panel_sizer.Add(self.load_image_button, 1, wx.EXPAND)
            self.load_image_button.Bind(wx.EVT_BUTTON, self.load_image)

            self.input_panel_sizer.Fit(self.input_panel)

        if True:
            self.pose_converter.init_pose_converter_panel(self.animation_panel)

        if True:
            self.animation_left_panel = wx.Panel(self.animation_panel, style=wx.SIMPLE_BORDER)
            self.animation_left_panel_sizer = wx.BoxSizer(wx.VERTICAL)
            self.animation_left_panel.SetSizer(self.animation_left_panel_sizer)
            self.animation_left_panel.SetAutoLayout(1)
            self.animation_panel_sizer.Add(self.animation_left_panel, 0, wx.EXPAND)

            self.result_image_panel = wx.Panel(self.animation_left_panel, size=(image_size, image_size),
                                               style=wx.SIMPLE_BORDER)
            self.result_image_panel.Bind(wx.EVT_PAINT, self.paint_result_image_panel)
            self.result_image_panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
            self.animation_left_panel_sizer.Add(self.result_image_panel, 0, wx.FIXED_MINSIZE)

            separator = wx.StaticLine(self.animation_left_panel, -1, size=(256, 5))
            self.animation_left_panel_sizer.Add(separator, 0, wx.EXPAND)

            background_text = wx.StaticText(self.animation_left_panel, label="--- Background ---",
                                            style=wx.ALIGN_CENTER)
            self.animation_left_panel_sizer.Add(background_text, 0, wx.EXPAND)

            self.output_background_choice = wx.Choice(
                self.animation_left_panel,
                choices=[
                    "TRANSPARENT",
                    "GREEN",
                    "BLUE",
                    "BLACK",
                    "WHITE"
                ])
            self.output_background_choice.SetSelection(0)
            self.animation_left_panel_sizer.Add(self.output_background_choice, 0, wx.EXPAND)

            separator = wx.StaticLine(self.animation_left_panel, -1, size=(256, 5))
            self.animation_left_panel_sizer.Add(separator, 0, wx.EXPAND)

            self.fps_text = wx.StaticText(self.animation_left_panel, label="")
            self.animation_left_panel_sizer.Add(self.fps_text, wx.SizerFlags().Border())

            self.animation_left_panel_sizer.Fit(self.animation_left_panel)

        self.animation_panel_sizer.Fit(self.animation_panel)

    def create_ui(self):
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.main_sizer)
        self.SetAutoLayout(1)

        self.capture_pose_lock = threading.Lock()

        self.create_connection_panel(self)
        self.main_sizer.Add(self.connection_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))

        self.create_animation_panel(self)
        self.main_sizer.Add(self.animation_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))

        self.create_capture_panel(self)
        self.main_sizer.Add(self.capture_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))

        self.main_sizer.Fit(self)

    def create_connection_panel(self, parent):
        self.connection_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.connection_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.connection_panel.SetSizer(self.connection_panel_sizer)
        self.connection_panel.SetAutoLayout(1)

        capture_device_ip_text = wx.StaticText(self.connection_panel, label="Capture Device IP:", style=wx.ALIGN_RIGHT)
        self.connection_panel_sizer.Add(capture_device_ip_text, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))

        self.capture_device_ip_text_ctrl = wx.TextCtrl(self.connection_panel, value="192.168.10.113")
#        self.capture_device_ip_text_ctrl = wx.TextCtrl(self.connection_panel, value="192.168.11.7")
        self.connection_panel_sizer.Add(self.capture_device_ip_text_ctrl, wx.SizerFlags(1).Expand().Border(wx.ALL, 3))

        self.start_capture_button = wx.Button(self.connection_panel, label="START CAPTURE!")
        self.connection_panel_sizer.Add(self.start_capture_button, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.start_capture_button.Bind(wx.EVT_BUTTON, self.on_start_capture)

    def create_capture_panel(self, parent):
        self.capture_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.capture_panel_sizer = wx.FlexGridSizer(cols=5)
        for i in range(5):
            self.capture_panel_sizer.AddGrowableCol(i)
        self.capture_panel.SetSizer(self.capture_panel_sizer)
        self.capture_panel.SetAutoLayout(1)

        self.rotation_labels = {}
        self.rotation_value_labels = {}
        rotation_column_0 = self.create_rotation_column(self.capture_panel, RIGHT_EYE_BONE_ROTATIONS)
        self.capture_panel_sizer.Add(rotation_column_0, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))
        rotation_column_1 = self.create_rotation_column(self.capture_panel, LEFT_EYE_BONE_ROTATIONS)
        self.capture_panel_sizer.Add(rotation_column_1, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))
        rotation_column_2 = self.create_rotation_column(self.capture_panel, HEAD_BONE_ROTATIONS)
        self.capture_panel_sizer.Add(rotation_column_2, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))

    def create_rotation_column(self, parent, rotation_names):
        column_panel = wx.Panel(parent, style=wx.SIMPLE_BORDER)
        column_panel_sizer = wx.FlexGridSizer(cols=2)
        column_panel_sizer.AddGrowableCol(1)
        column_panel.SetSizer(column_panel_sizer)
        column_panel.SetAutoLayout(1)

        for rotation_name in rotation_names:
            self.rotation_labels[rotation_name] = wx.StaticText(
                column_panel, label=rotation_name, style=wx.ALIGN_RIGHT)
            column_panel_sizer.Add(self.rotation_labels[rotation_name],
                                   wx.SizerFlags(1).Expand().Border(wx.ALL, 3))

            self.rotation_value_labels[rotation_name] = wx.TextCtrl(
                column_panel, style=wx.TE_RIGHT)
            self.rotation_value_labels[rotation_name].SetValue("0.00")
            self.rotation_value_labels[rotation_name].Disable()
            column_panel_sizer.Add(self.rotation_value_labels[rotation_name],
                                   wx.SizerFlags(1).Expand().Border(wx.ALL, 3))

        column_panel.GetSizer().Fit(column_panel)
        return column_panel

    def paint_capture_panel(self, event: wx.Event):
        self.update_capture_panel(event)

    def update_capture_panel(self, event: wx.Event):
        data = self.ifacialmocap_pose
        for rotation_name in ROTATION_NAMES:
            value = data[rotation_name]
            self.rotation_value_labels[rotation_name].SetValue("%0.2f" % value)

    @staticmethod
    def convert_to_100(x):
        return int(max(0.0, min(1.0, x)) * 100)

    def paint_source_image_panel(self, event: wx.Event):
        wx.BufferedPaintDC(self.source_image_panel, self.source_image_bitmap)

    def update_source_image_bitmap(self):
        dc = wx.MemoryDC()
        dc.SelectObject(self.source_image_bitmap)
        if self.wx_source_image is None:
            self.draw_nothing_yet_string(dc)
        else:
            dc.Clear()
            dc.DrawBitmap(self.wx_source_image, 0, 0, True)
        del dc

    def draw_nothing_yet_string(self, dc):
        dc.Clear()
        font = wx.Font(wx.FontInfo(14).Family(wx.FONTFAMILY_SWISS))
        dc.SetFont(font)
        w, h = dc.GetTextExtent("Nothing yet!")
        dc.DrawText("Nothing yet!", (self.poser.get_image_size() - w) // 2, (self.poser.get_image_size() - h) // 2)

    def paint_result_image_panel(self, event: wx.Event):
        wx.BufferedPaintDC(self.result_image_panel, self.result_image_bitmap)

    def update_result_image_bitmap(self, event: Optional[wx.Event] = None):
        """
        XXX: ここがメインロジック
        """
        start_time = time.time_ns()
        ifacialmocap_pose = self.read_ifacialmocap_pose()
        current_pose = self.pose_converter.convert(ifacialmocap_pose)
        if self.last_pose is not None and self.last_pose == current_pose:
            return
        self.last_pose = current_pose

        if self.torch_source_image is None:
            dc = wx.MemoryDC()
            dc.SelectObject(self.result_image_bitmap)
            self.draw_nothing_yet_string(dc)
            del dc
            return

        pose = torch.tensor(current_pose, device=self.device, dtype=self.poser.get_dtype())
        pose_time = time.time_ns()
        with torch.no_grad():
            output_image = self.poser.pose(self.torch_source_image, pose)[0].float()
            output_image = convert_linear_to_srgb((output_image + 1.0) / 2.0)

            background_choice = self.output_background_choice.GetSelection()
            if background_choice == 0:
                pass
            else:
                background = torch.zeros(4, output_image.shape[1], output_image.shape[2], device=self.device)
                background[3, :, :] = 1.0
                if background_choice == 1:
                    background[1, :, :] = 1.0
                    output_image = self.blend_with_background(output_image, background)
                elif background_choice == 2:
                    background[2, :, :] = 1.0
                    output_image = self.blend_with_background(output_image, background)
                elif background_choice == 3:
                    output_image = self.blend_with_background(output_image, background)
                else:
                    background[0:3, :, :] = 1.0
                    output_image = self.blend_with_background(output_image, background)

            c, h, w = output_image.shape
            output_image = 255.0 * torch.transpose(output_image.reshape(c, h * w), 0, 1).reshape(h, w, c)
            output_image = output_image.byte()

        torch_time = time.time_ns()
        # XXX fifoを使ってRVCに合わせた遅延をいれる
        numpy_image_ = output_image.detach().cpu().numpy()
        self.fifo_array.push(numpy_image_)
        numpy_image = self.fifo_array.pop()

#        numpy_image = numpy_image_ # XXX 実験のため一時的に遅延なし

        wx_image = wx.ImageFromBuffer(numpy_image.shape[0],
                                      numpy_image.shape[1],
                                      numpy_image[:, :, 0:3].tobytes(),
                                      numpy_image[:, :, 3].tobytes())
        wx_bitmap = wx_image.ConvertToBitmap()

        dc = wx.MemoryDC()
        dc.SelectObject(self.result_image_bitmap)
        dc.Clear()
        dc.DrawBitmap(wx_bitmap,
                      (self.poser.get_image_size() - numpy_image.shape[0]) // 2,
                      (self.poser.get_image_size() - numpy_image.shape[1]) // 2, True)
        del dc

        end_time = time.time_ns()
        if self.last_update_time is not None:
            call_interval_time = end_time - self.last_update_time
            fps = 1.0 / (call_interval_time / 10**9)
            if self.torch_source_image is not None:
                self.statistics.add("fps", fps)
            FPS = self.statistics.get("fps")
            txt = f"FPS={FPS:.2f} "
            txt += self.statistics.txt("interval", call_interval_time)
            txt += self.statistics.txt("all", end_time - start_time)
            txt += self.statistics.txt("pose", pose_time - start_time)
            txt += self.statistics.txt("img", torch_time - pose_time)
            txt += self.statistics.txt("bitmap", end_time - torch_time)
            self.fps_text.SetLabelText(txt)
        self.last_update_time = end_time

        self.Refresh()

    def blend_with_background(self, numpy_image, background):
        alpha = numpy_image[3:4, :, :]
        color = numpy_image[0:3, :, :]
        new_color = color * alpha + (1.0 - alpha) * background[0:3, :, :]
        return torch.cat([new_color, background[3:4, :, :]], dim=0)

    def load_image_direct(self, image_file_name: str) -> None:
        try:
            pil_image = resize_PIL_image(
                extract_PIL_image_from_filelike(image_file_name),
                (self.poser.get_image_size(), self.poser.get_image_size()))
            w, h = pil_image.size
            if pil_image.mode != 'RGBA':
                self.source_image_string = "Image must have alpha channel!"
                self.wx_source_image = None
                self.torch_source_image = None
            else:
                self.wx_source_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                self.torch_source_image = extract_pytorch_image_from_PIL_image(pil_image) \
                    .to(self.device).to(self.poser.get_dtype())
            self.update_source_image_bitmap()
        except:
            message_dialog = wx.MessageDialog(self, "Could not load image " + image_file_name, "Poser", wx.OK)
            message_dialog.ShowModal()
            message_dialog.Destroy()

    def load_image(self, event: wx.Event):
        dir_name = "data/images"
        file_dialog = wx.FileDialog(self, "Choose an image", dir_name, "", "*.png", wx.FD_OPEN)
        if file_dialog.ShowModal() == wx.ID_OK:
            image_file_name = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
            try:
                pil_image = resize_PIL_image(
                    extract_PIL_image_from_filelike(image_file_name),
                    (self.poser.get_image_size(), self.poser.get_image_size()))
                w, h = pil_image.size
                if pil_image.mode != 'RGBA':
                    self.source_image_string = "Image must have alpha channel!"
                    self.wx_source_image = None
                    self.torch_source_image = None
                else:
                    self.wx_source_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                    self.torch_source_image = extract_pytorch_image_from_PIL_image(pil_image) \
                        .to(self.device).to(self.poser.get_dtype())
                self.update_source_image_bitmap()
            except:
                message_dialog = wx.MessageDialog(self, "Could not load image " + image_file_name, "Poser", wx.OK)
                message_dialog.ShowModal()
                message_dialog.Destroy()
        file_dialog.Destroy()
        self.Refresh()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Control characters with movement captured by CuTalk.')
    parser.add_argument(
        '--model',
        type=str,
        required=False,
        default='standard_float',
        choices=['standard_float', 'separable_float', 'standard_half', 'separable_half'],
        help='The model to use.')
    args = parser.parse_args()

    device = torch.device('cuda')
    try:
        poser = load_poser(args.model, device)
    except RuntimeError as e:
        print(e)
        sys.exit()

    from tha3.mocap.ifacialmocap_poser_converter_25 import create_ifacialmocap_pose_converter

    pose_converter = create_ifacialmocap_pose_converter()

    app = wx.App()
    main_frame = MainFrame(poser, pose_converter, device)
    main_frame.Show(True)
    main_frame.capture_timer.Start(10) #ms
    main_frame.animation_timer.Start(10)
    app.MainLoop()
