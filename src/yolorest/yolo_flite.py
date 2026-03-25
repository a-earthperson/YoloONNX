# Based on the code from Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import logging
import os
import time

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

from yolorest.prediction import Prediction, Predictions

logger = logging.getLogger(__name__)


class YOLOFLite:
    """
    A class for performing object detection using the YOLOv8 model with TensorFlow Lite.

    This class handles model loading, preprocessing, inference, and visualization of detection results.

    Attributes:
        model (Interpreter): TensorFlow Lite interpreter for the YOLOv8 model.
        labels (Dict[int, Label]): Dictionary mapping class IDs to Label names.
        conf (float): Confidence threshold for filtering detections.
        iou (float): Intersection over Union threshold for non-maximum suppression.
        color_palette (np.ndarray): Random color palette for visualization with shape (num_labels, 3).
        in_width (int): Input width required by the model.
        in_height (int): Input height required by the model.
        in_index (int): Input tensor index in the model.
        in_scale (float): Input quantization scale factor.
        in_zero_point (int): Input quantization zero point.
        int8 (bool): Whether the model uses int8 quantization.
        out_index (int): Output tensor index in the model.
        out_scale (float): Output quantization scale factor.
        out_zero_point (int): Output quantization zero point.

    Methods:
        letterbox: Resizes and pads image while maintaining aspect ratio.
        draw_detections: Draws bounding boxes and labels on the input image.
        preprocess: Preprocesses the input image before inference.
        postprocess: Processes model outputs to extract and visualize detections.
        detect: Performs object detection on an input image.
    """

    # todo: change metadata to labels, that is a dict int, string, support both yaml and labelmap https://github.com/google-coral/tflite/blob/eced31ac01e9c2636150decef7d3c335d0feb304/python/examples/classification/classify_image.py#L55
    def __init__(
        self,
        model: str,
        labels: dict[int, str],
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
    ):
        """
        Initialize an instance of the YOLOv8TFLite class.

        Args:
            model (str): Path to the TFLite model file.
            labels: (Dict[int, Label]):  Dictionary mapping class IDs to Label names.
            conf (float): Confidence threshold for filtering detections.
            iou (float): IoU threshold for non-maximum suppression.
            device (str): auto, cpu, usb, usb:0, usb:1, pci:1, pci:2
        """
        self.conf = conf
        self.iou = iou
        self.labels = labels

        np.random.seed(42)  # Set seed for reproducible colors
        self.color_palette = np.random.uniform(128, 255, size=(len(self.labels), 3))

        logger.info(f"Attempting to load TPU as {device}")
        if device == "cpu":
            self.model = Interpreter(model_path=model)
        else:
            try:
                device_config = {"device": device}
                interpreter_delegate = load_delegate("libedgetpu.so.1.0", device_config)
                logger.info("TPU found")
                self.model = Interpreter(
                    model_path=model,
                    experimental_delegates=[interpreter_delegate],
                )
            except ValueError:
                _, ext = os.path.splitext(model)

                if ext and ext != ".tflite":
                    logger.error(
                        "Incorrect model used with EdgeTPU. Only .tflite models can be used with a Coral EdgeTPU or CPU."
                    )
                else:
                    logger.error(
                        "No EdgeTPU was detected. If you do not have a Coral device yet, you must configure CPU detectors."
                    )

                raise

        self.model.allocate_tensors()

        input_details = self.model.get_input_details()[0]
        self.in_width, self.in_height = input_details["shape"][1:3]
        self.in_index = input_details["index"]
        self.in_scale, self.in_zero_point = input_details["quantization"]
        self.int8 = input_details["dtype"] == np.int8

        output_details = self.model.get_output_details()[0]
        self.out_index = output_details["index"]
        self.out_scale, self.out_zero_point = output_details["quantization"]

    def letterbox(
        self, img: np.ndarray, new_shape: tuple[int, int] = (640, 640)
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """
        Resize and pad image while maintaining aspect ratio.

        Args:
            img (np.ndarray): Input image with shape (H, W, C).
            new_shape (Tuple[int, int]): Target shape (height, width).

        Returns:
            (np.ndarray): Resized and padded image.
            (Tuple[float, float]): Padding ratios (top/height, left/width) for coordinate adjustment.
        """
        start_time = time.time()
        shape = img.shape[:2]

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        duration_ms = (time.time() - start_time) * 1000
        logger.debug(f"letterbox took {duration_ms:.2f} ms")

        return img, (top / img.shape[0], left / img.shape[1])

    def draw_detections(self, img: np.ndarray, predictions: Predictions) -> None:
        """
        Draw bounding boxes and labels on the input image based on the detected objects.

        Args:
            img (np.ndarray): The input image to draw detections on.
            detections: List[Detection]: List of detected objects with their bounding boxes, scores, and class IDs.
        """
        font_scale = 10
        for prediction in predictions.predictions:
            color = self.color_palette[prediction.label.id]
            cv2.rectangle(
                img,
                (int(prediction.x_min), int(prediction.y_min)),
                (int(prediction.x_max), int(prediction.y_max)),
                color,
                2,
            )
            label = f"{prediction.label.name}: {prediction.score:.2f}%"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            label_x = prediction.x_min
            label_y = (
                prediction.y_min - 10
                if prediction.y_min - 10 > label_height
                else prediction.y_min + 10
            )
            cv2.rectangle(
                img,
                (int(label_x), int(label_y - label_height)),
                (int(label_x + label_width), int(label_y + label_height)),
                color,
                cv2.FILLED,
            )

            cv2.putText(
                img,
                label,
                (int(label_x), int(label_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """
        Preprocess the input image before performing inference.

        Args:
            img (np.ndarray): The input image to be preprocessed with shape (H, W, C).

        Returns:
            (np.ndarray): Preprocessed image ready for model input.
            (Tuple[float, float]): Padding ratios for coordinate adjustment.
        """
        img, pad = self.letterbox(img, (self.in_width, self.in_height))
        img = img[..., ::-1][None]
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        return img / 255, pad

    def postprocess(
        self, img: np.ndarray, outputs: np.ndarray, pad: tuple[float, float]
    ) -> Predictions:
        """
        Process model outputs to extract and visualize detections.

        Args:
            img (np.ndarray): The original input image.
            outputs (np.ndarray): Raw model outputs.
            pad (Tuple[float, float]): Padding ratios from preprocessing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]
        outputs[:, :4] *= max(img.shape)

        outputs = outputs.transpose(0, 2, 1)
        outputs[..., 0] -= outputs[..., 2] / 2
        outputs[..., 1] -= outputs[..., 3] / 2
        predictions = Predictions(predictions=[], success=True)

        for out in outputs:
            scores = out[:, 4:].max(-1)
            keep = scores > self.conf
            if not keep.any():
                logger.debug("No detections passed the confidence threshold.")
                continue
            boxes = out[keep, :4]
            scores = scores[keep]
            class_ids = out[keep, 4:].argmax(-1)
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou).flatten()
            for i in indices:
                label = self.labels[class_ids[i]]
                score = scores[i]
                box = boxes[i]
                left, top, w, h = box
                right = left + w
                bottom = top + h
                predictions.predictions.append(
                    Prediction(
                        label=label,
                        confidence=score,
                        y_min=top,
                        x_min=left,
                        y_max=bottom,
                        x_max=right,
                    )
                )
        return predictions

    def detect(self, img: np.ndarray) -> Predictions:
        """
        Perform object detection on an input image.

        Args:
            img (np.ndarray): Image

        Returns:
            (List[Detection]): List of detected objects with their bounding boxes, scores and label.
        """
        x, pad = self.preprocess(img)

        if self.int8:
            x = (x / self.in_scale + self.in_zero_point).astype(np.int8)

        start_time = time.time()
        self.model.set_tensor(self.in_index, x)
        self.model.invoke()
        duration_ms = (time.time() - start_time) * 1000
        logger.debug(f"detect took {duration_ms:.2f} ms")

        y = self.model.get_tensor(self.out_index)
        if self.int8:
            y = (y.astype(np.float32) - self.out_zero_point) * self.out_scale

        return self.postprocess(img, y, pad)
