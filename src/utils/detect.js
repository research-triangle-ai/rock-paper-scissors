import { Tensor } from "onnxruntime-web";
import { renderBoxes } from "./renderBox";

/**
 * Detect Image
 * @param {String} image Image URL
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session YOLOv10 onnxruntime session
 * @param {Number} scoreThreshold Float representing the threshold for deciding when to remove boxes based on score
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 * @param {HTMLElement} timeRef
 */
export const detectImage = async (
  image,
  canvas,
  session,
  scoreThreshold,
  modelInputShape,
  timeElement,
  callback
) => {
  // clean up canvas
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clear canvas

  let imageT;
  try {
    // Try to create tensor from image
    imageT = await Tensor.fromImage(image, { tensorFormat: "RGB" });
  } catch (error) {
    console.error("Error creating tensor from image:", error);
    // If tensor creation fails, try to draw the image on canvas and create tensor from that
    const img = new Image();
    img.src = image;
    await new Promise((resolve) => {
      img.onload = resolve;
    });
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    imageT = new Tensor("uint8", new Uint8Array(imageData.data), [1, canvas.height, canvas.width, 4]);
  }

  const [modelHeight, modelWidth] = modelInputShape.slice(2);
  const [imgHeight, imgWidth] = imageT.dims.slice(2);

  // Padding
  const max_ = Math.max(imgWidth, imgHeight);
  const padWidth = max_ - imgWidth;
  const padHeight = max_ - imgHeight;
  const padL = Math.floor(padWidth / 2);
  const padR = padWidth - padL;
  const padU = Math.floor(padHeight / 2);
  const padB = padHeight - padU;

  const padding = new Tensor(
    "int64",
    new BigInt64Array([padU, padL, padB, padR].map((e) => BigInt(e))) // [up, left, bottom, right]
  );

  // Resizing
  const scaleH = modelHeight / max_;
  const scaleW = modelWidth / max_;

  const scales = new Tensor(
    "float32",
    new Float32Array([scaleH, scaleW]) // [sh, sw]
  );

  const start = Date.now();
  const { letterbox } = await session.prep.run({
    images: imageT,
    padding: padding,
    scales: scales,
  }); // run preprocessing, padding and resize

  const { output0 } = await session.net.run({ images: letterbox }); // run session and get output layer
  // const inferenceTime = Date.now() - start; // This is now a number in milliseconds

  // if (timeElement) {
  //   timeElement.innerHTML = `Inference time: ${inferenceTime.toFixed(2)} ms`;
  // }

  const boxes = [];
  var detection;
  let hasValidDetection = false;

  // looping through output
  for (let idx = 0; idx < output0.dims[1]; idx++) {
    const [left, top, right, bottom, conf, cid] = output0.data.slice(
      idx * output0.dims[2],
      (idx + 1) * output0.dims[2]
    ); // get rows

    if (conf < scoreThreshold) break; // break if conf is lesser than threshold (because it's sorted)

    const [x, y, w, h] = [
      left / scaleW - padL, // upscale left
      top / scaleH - padU, // upscale top
      (right - left) / scaleW, // upscale width
      (bottom - top) / scaleH, // upscale height
    ]; // keep boxes in maxSize range

    boxes.push({
      label: cid,
      probability: conf,
      bounding: [x, y, w, h], // upscale box
    }); // update boxes to draw later

    hasValidDetection = true;
    detection = {x, y, width: w, height: h, label: cid};
    break;
  }

  // rendering result
  // set canvas res the same as image res
  ctx.canvas.width = imgWidth;
  ctx.canvas.height = imgHeight;

  ctx.putImageData(await imageT.toImageData(), 0, 0); // Draw image
  renderBoxes(ctx, boxes); // Draw boxes

  // Call the callback with the detection result
  if (callback) {
    callback(hasValidDetection, detection);
  }
};

// Remove or comment out the normalizeTime function as it's no longer used
// const normalizeTime = (time) => {
//   if (time < 1000) return `${time} ms`;
//   else if (time < 60000) return `${(time / 1000).toFixed(2)} S`;
//   return `${(time / 60000).toFixed(2)} H`;
// };