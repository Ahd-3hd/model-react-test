import * as tf from "@tensorflow/tfjs";
import { useCallback, useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";

const classes: string[] = [
  "Air intake",
  "Console",
  "Dashboard",
  "Fog light",
  "Gear stick",
  "Headlight",
  "No car",
  "Steering wheel",
  "Tail light",
];

// const predict = async (model: tf.GraphModel, myimg: File) => {
//   const imgEle = new Image();
//   imgEle.src = URL.createObjectURL(myimg);

//   const imageTensor = tf.browser.fromPixels(imgEle);

//   const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]);
//   const preprocessedImage = resizedImage.toFloat().div(tf.scalar(255));
//   const inputTensor = preprocessedImage.expandDims(0);
//   const predictions = model.predict(inputTensor) as unknown as tf.Tensor;

//   const predictedLabel = tf.argMax(predictions, 1).dataSync()[0];
//   const probs = tf.softmax(predictions, 1);
//   const maxProb = tf.max(probs, 1);
//   const maxProbValue = (maxProb as any).arraySync()[0];

//   console.log(predictedLabel);

//   // tf.tidy(() => {
//   //   const img = tf.cast(tf.browser.fromPixels(imgEle), "float32");
//   //   const batched = img.expandDims(0);
//   //   const prediction = model.predict(batched) as unknown as tf.Tensor;
//   //   const confidenceScores = prediction.squeeze().arraySync() as number[];
//   //   return confidenceScores.map((confidence, index) => ({
//   //     confidence,
//   //     name: classes[index],
//   //   }));
//   // });
// };

function App() {
  const webcamRef = useRef<any>(null);

  const [devices, setDevices] = useState([]);

  const [label, setLabel] = useState("");

  const handleDevices = useCallback(
    (mediaDevices: any) =>
      setDevices(
        mediaDevices.filter(
          ({ kind }: { kind: string }) => kind === "videoinput"
        )
      ),
    [setDevices]
  );

  console.log(devices);

  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then(handleDevices);
  }, [handleDevices]);

  const [videoConstraints, setVideoConstraints] = useState({
    width: 224,
    height: 224,
    facingMode: "environment",
    deviceId:
      "915c707532ac5d746f9b48edc5caa22291512103faa7568b5417800a6e0d9a97",
  } as {
    width: number;
    height: number;
    facingMode: string;
    deviceId?: string;
  });

  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [error, setError] = useState(null);

  const loadModel = async () => {
    try {
      const model = await tf.loadGraphModel("./model.json");
      console.log("loaded");
      setModel(model as any);
    } catch (error) {
      console.log(error);
    }
  };

  const capture = useCallback(() => {
    if (!webcamRef.current) return;

    const imageSrc = webcamRef.current.getScreenshot();
    const imageElement = new Image();
    imageElement.src = imageSrc;

    // console.log(imageSrc);

    imageElement.onload = () => {
      // Create a canvas element
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      // Set the canvas dimensions to match the image
      canvas.width = imageElement.width;
      canvas.height = imageElement.height;

      if (!ctx) return;
      if (!model) return;
      // Draw the image on the canvas
      ctx.drawImage(imageElement, 0, 0);

      // Get the image data from the canvas
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const img = tf.cast(tf.browser.fromPixels(imageData), "float32");
      const inputTensor = img.reshape([1, 224, 224, 3]);
      // Make predictions using the image tensor
      const predictions = model.predict(inputTensor) as unknown as tf.Tensor;
      const predictedLabel = tf.argMax(predictions, 1).dataSync()[0];

      console.log(predictedLabel, classes[predictedLabel]);
      setLabel(classes[predictedLabel]);

      // Further process or analyze the predictions
      // ...
    };

    // getPrediction(imageSrc);
  }, [model]);

  useEffect(() => {
    tf.ready()
      .then(() => {
        loadModel();
      })
      .catch((err) => setError(err));
  }, []);

  useEffect(() => {
    const x = setInterval(() => {
      capture();
    }, 10);
    return () => {
      clearInterval(x);
    };
  }, [capture]);

  if (error) return <pre>{JSON.stringify(error)}</pre>;
  if (!model) return <h1>Loading</h1>;

  return (
    <>
      <Webcam
        ref={webcamRef}
        videoConstraints={videoConstraints}
        screenshotFormat="image/jpeg"
        screenshotQuality={1}
      />
      <button onClick={capture}>predict</button>
      <h1>{label}</h1>
      {/* 
      <Webcam
        ref={webcamRef}
        videoConstraints={{
          ...videoConstraints,
          width: 600,
          height: 600,
        }}
        screenshotFormat="image/jpeg"
        screenshotQuality={1}
      /> */}
    </>
  );
}

export default App;
