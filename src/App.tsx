import * as tf from "@tensorflow/tfjs";
import { useCallback, useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";

const imagesPaths: Record<string, string> = {
  "Air intake": "/air_intake.jpeg",
  Console: "/console.jpeg",
  Dashboard: "/dashboard2.jpg",
  "Fog light": "/foglight.jpeg",
  "Gear stick": "/gearstick.jpg",
  Headlight: "/headlight.jpg",
  "Steering wheel": "/steeringwheel.jpg",
  "Tail light": "/taillight.jpg",
};

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
  const webcamRef2 = useRef<any>(null);

  const [label, setLabel] = useState("");

  const [currentImage, setCurrentImage] = useState("");

  const [info, setInfo] = useState(false);

  // const [devices, setDevices] = useState([]);
  // const handleDevices = useCallback(
  //   (mediaDevices: any) =>
  //     setDevices(
  //       mediaDevices.filter(
  //         ({ kind }: { kind: string }) => kind === "videoinput"
  //       )
  //     ),
  //   [setDevices]
  // );

  // console.log(devices);

  // useEffect(() => {
  //   navigator.mediaDevices.enumerateDevices().then(handleDevices);
  // }, [handleDevices]);

  const [videoConstraints, setVideoConstraints] = useState({
    width: 224,
    height: 224,
    facingMode: "environment",
    // deviceId:
    //   "fee45256aec9e52564209e6c4d3607cd6ad2cdee607ff5f92a7d514348198291",
  } as {
    width: number;
    height: number;
    facingMode: string;
    deviceId?: string;
  });

  const [videoConstraints2, setVideoConstraints2] = useState({
    width: window.visualViewport?.width ?? window.innerWidth ?? 224,
    height: window.visualViewport?.height ?? window.innerHeight ?? 224,
    facingMode: "environment",
    // deviceId:
    //   "fee45256aec9e52564209e6c4d3607cd6ad2cdee607ff5f92a7d514348198291",
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
      try {
        tf.tidy(() => {
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
          const predictions = model.predict(
            inputTensor
          ) as unknown as tf.Tensor;
          const predictedLabel = tf.argMax(predictions, 1).dataSync()[0];
          // console.log(predictedLabel);
          setLabel(classes[predictedLabel]);
        });
      } catch (error) {
        console.log(error);
      }
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

  useEffect(() => {
    const ww = document.documentElement.clientWidth;
    const hh = document.documentElement.clientHeight;

    setVideoConstraints2((prev) => ({
      ...prev,
      width: ww ?? window.visualViewport?.width ?? window.innerWidth ?? 224,
      height: hh ?? window.visualViewport?.height ?? window.innerHeight ?? 224,
      // deviceId:
      //   '255d509c830c05bd31f8a48aa1c4340345324d56fbd574f49cc2c55d71d96d40',
    }));

    window.addEventListener("load", () => {
      setVideoConstraints2((prev) => ({
        ...prev,
        width: ww ?? window.visualViewport?.width ?? window.innerWidth ?? 224,
        height:
          hh ?? window.visualViewport?.height ?? window.innerHeight ?? 224,
      }));
    });

    window.addEventListener("resize", () => {
      setVideoConstraints2((prev) => ({
        ...prev,
        width: ww ?? window.visualViewport?.width ?? window.innerWidth ?? 224,
        height:
          hh ?? window.visualViewport?.height ?? window.innerHeight ?? 224,
      }));
    });
  }, []);

  if (error)
    return <pre>this is an error please refresh - {JSON.stringify(error)}</pre>;
  if (!model)
    return (
      <h1>
        Loading ... depends on your connection takes about 2-3 mins to load
      </h1>
    );

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        width: "100vw",
        height: "100vh",
        overflow: "hidden",
      }}
    >
      <button
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          background: "black",
          border: "none",
          fontWeight: "bold",
          color: "white",
          margin: "0.4rem",
          fontSize: "1rem",
          zIndex: 100,
        }}
        onClick={() => {
          setInfo(true);
        }}
      >
        Click info
      </button>

      {info && (
        <div
          style={{
            position: "fixed",
            height: "100vh",
            textAlign: "center",
            background: "#183d75",
            color: "white",
            padding: "0.5rem",
            margin: "0 auto",
            zIndex: 100,
          }}
        >
          <p>This is a model that detects 8 parts of a car.</p>
          <p>Click to see how they look like:</p>
          <p>
            don't let it running in the background, it will drain your battery.
          </p>

          <div
            style={{
              width: "80%",
              textAlign: "center",
              margin: "0 auto",
            }}
          >
            {classes
              .filter((c) => c !== "No car")
              .map((c) => (
                <button
                  key={c}
                  style={{
                    background: "black",
                    border: "none",
                    fontWeight: "bold",
                    color: "white",
                    margin: "0.4rem",
                    fontSize: "1rem",
                  }}
                  onClick={() => {
                    setCurrentImage(imagesPaths[c]);
                  }}
                >
                  {c}
                </button>
              ))}

            <div>
              <img src={currentImage} alt="" width="260px" />
            </div>

            <button
              onClick={() => {
                setCurrentImage("");
                setInfo(false);
              }}
            >
              GO BACK
            </button>
          </div>
        </div>
      )}
      <Webcam
        ref={webcamRef2}
        videoConstraints={{
          ...videoConstraints2,
          // deviceId:
          //   "fee45256aec9e52564209e6c4d3607cd6ad2cdee607ff5f92a7d514348198291",
        }}
        screenshotFormat="image/jpeg"
        screenshotQuality={0.8}
      />
      <Webcam
        style={{
          opacity: 0,
          visibility: "hidden",
          position: "fixed",
        }}
        ref={webcamRef}
        videoConstraints={{
          ...videoConstraints,
          // deviceId:
          //   "fee45256aec9e52564209e6c4d3607cd6ad2cdee607ff5f92a7d514348198291",
        }}
        screenshotFormat="image/jpeg"
        screenshotQuality={0.8}
        forceScreenshotSourceSize
      />

      {label && (
        <h3
          key={label}
          style={{
            position: "fixed",
            bottom: "1rem",
            left: "50%",
            width: "80%",
            textAlign: "center",
            background: "#183d75",
            color: "white",
            padding: "0.5rem",
            margin: "0 auto",
            transform: "translateX(-50%)",
          }}
        >
          {label}
        </h3>
      )}
    </div>
  );
}

export default App;
