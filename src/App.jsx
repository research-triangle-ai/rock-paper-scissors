import { useState, useEffect, useRef } from "react";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import { download } from "./utils/download";
import "./style/App.css";

const App = () => {
  const [session, setSession] = useState(null);
  const [games, setGames] = useState([]);
  const [lastPlayerMove, setLastPlayerMove] = useState(null);
  const [lastComputerMove, setLastComputerMove] = useState(1); // Start with paper
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState({
    text: "",
    progress: null,
  });
  const [objectDetected, setObjectDetected] = useState(false);
  const timeRef = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const captureIntervalRef = useRef(null);
  const previousImageDataRef = useRef(null);
  const stableFrameCountRef = useRef(0);
  const [isWebcamRunning, setIsWebcamRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [videoZIndex, setVideoZIndex] = useState(3); // Default zIndex

  // Configs
  const modelName = "rps.onnx";
  const modelInputShape = [1, 3, 640, 640];
  const scoreThreshold = 0.75;
  const captureInterval = 4;
  const stableFrameThreshold = 5; // Number of stable frames to consider movement stopped
  const movementThreshold = 0.010; // Extracted and increased threshold for movement detection

  useEffect(() => {
    const init = async () => {
      const baseModelURL = `model`;

      // create session
      const arrBufNet = await download(
        `${baseModelURL}/${modelName}`, // url
        ["Loading YOLOv10 Rock/Paper/Scissors Detection model", setLoading] // logger
      );
      const yolov10 = await InferenceSession.create(arrBufNet);
      const prepBuf = await download(
        `${baseModelURL}/preprocess-yolo.onnx`, // url
        ["Loading Preprocessing model", setLoading] // logger
      );
      const prep = await InferenceSession.create(prepBuf);

      // warmup main model
      setLoading({ text: "Warming up model...", progress: null });
      const tensor = new Tensor(
        "float32",
        new Float32Array(modelInputShape.reduce((a, b) => a * b)),
        modelInputShape
      );
      await yolov10.run({ images: tensor });

      setSession({ net: yolov10, prep: prep });
      setLoading(null);
    }

    init();
  }, []);

  useEffect(() => {
    if (!isPaused) {
      // console.log('isPaused is false, starting capture interval');
      startCaptureInterval();
    } else {
      // console.log('isPaused is true, stopping capture interval');
      clearInterval(captureIntervalRef.current);
    }

    const handleKeyPress = (event) => {
      if (event.code === 'Space' && isPaused) {
        resumeCapture();
      }
    };

    window.addEventListener('keydown', handleKeyPress);

    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [isPaused]);

  const addGameResult = (playerMove, playerImage) => {
    let computerMove;

    if (games.length === 0) {
      // First move: play paper
      computerMove = 1;
    } else {
      const lastGame = games[games.length - 1];
      if ((lastGame.computer === 0 && lastGame.player === 2) ||
          (lastGame.computer === 1 && lastGame.player === 0) ||
          (lastGame.computer === 2 && lastGame.player === 1)) {
        // Computer won: play what the opponent just lost with
        computerMove = lastPlayerMove;
      } else if (lastGame.computer === lastGame.player) {
        // Tie: play what was not played last round
        computerMove = [0, 1, 2].find(move => move !== lastPlayerMove && move !== lastComputerMove);
      } else {
        // Computer lost: play what was not played last round
        computerMove = [0, 1, 2].find(move => move !== lastPlayerMove && move !== lastComputerMove);
      }
    }

    const newGame = {
      id: games.length,
      player: playerMove,
      computer: computerMove,
      playerImage: playerImage,
    };

    setGames([...games, newGame]);
    setLastPlayerMove(playerMove);
    setLastComputerMove(computerMove);
  };

  const calculateWinningRate = () => {
    const totalGames = games.length;
    const wins = games.reduce((acc, game) => {
      if ((game.player === 0 && game.computer === 2) ||
          (game.player === 1 && game.computer === 0) ||
          (game.player === 2 && game.computer === 1)) {
        return acc + 1; // Player wins
      } else if (game.player === game.computer) {
        return acc + 0.5; // Draw
      } else {
        return acc; // Player loses
      }
    }, 0);
    return totalGames > 0 ? (wins / totalGames) * 100 : 0;
  };

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          setIsWebcamRunning(true);
          startCaptureInterval();
        };
      }
    } catch (error) {
      console.error("Error accessing webcam:", error);
    }
  };

  const startCaptureInterval = () => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
    }
    captureIntervalRef.current = setInterval(() => {
      if (!isPaused) {
        captureAndCompare();
      }
    }, captureInterval);
  };

  const captureAndCompare = () => {
    // console.log("Capture and compare called");
    if (videoRef.current && videoRef.current.videoWidth > 0 && videoRef.current.videoHeight > 0) {
      // console.log("Video ref is valid");
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;

      const ctx = canvas.getContext('2d');
      ctx.save();

      // Flip the context horizontally to display the video feed correctly
      ctx.scale(-1, 1);
      ctx.translate(-canvas.width, 0);

      // Draw the video frame onto the flipped context
      ctx.drawImage(videoRef.current, 0, 0);
      ctx.restore();

      try {
        const imageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);

        if (previousImageDataRef.current) {
          const diff = compareImageData(previousImageDataRef.current, imageData);
          // console.log("Difference:", diff);
          if (diff < movementThreshold) {
            stableFrameCountRef.current++;
            if (stableFrameCountRef.current >= stableFrameThreshold) {
              const imageDataUrl = canvas.toDataURL('image/jpeg');
              // console.log("Stable frame count reached, setting imageDataUrl");
              setImage(imageDataUrl);
            }
          } else {
            // console.log("Movement detected, resetting stable frame count");
            stableFrameCountRef.current = 0;
          }
        } else {
          // console.log("previousImageDataRef.current is null");
        }

        previousImageDataRef.current = imageData;
      } catch (error) {
        console.error("Error capturing image:", error);
      }
    } else {
      // console.log("Video ref is not valid");
    }
  };

  const compareImageData = (data1, data2) => {
    let diff = 0;
    for (let i = 0; i < data1.data.length; i += 4) {
      diff += Math.abs(data1.data[i] - data2.data[i]);
    }
    return diff / (data1.width * data1.height * 255);
  };

  const handleDetectionResult = (hasDetections, detection) => {
    if (hasDetections && !objectDetected) {
      setObjectDetected(hasDetections);
      setIsPaused(true);
      clearInterval(captureIntervalRef.current);
      setVideoZIndex(0); // Send video to back during detection

      // Create a new canvas for drawing detection results
      const detectionCanvas = document.createElement('canvas');
      detectionCanvas.width = canvasRef.current.width;
      detectionCanvas.height = canvasRef.current.height;
      const detectionCtx = detectionCanvas.getContext('2d');

      // Draw the video frame onto the new canvas
      detectionCtx.drawImage(videoRef.current, 0, 0, detectionCanvas.width, detectionCanvas.height);

      // Draw detections on the new canvas
      const { x, y, width, height, label } = detection;

      // // Draw bounding box
      // detectionCtx.strokeStyle = 'red';
      // detectionCtx.lineWidth = 2;
      // detectionCtx.strokeRect(x, y, width, height);

      // // Draw label
      // detectionCtx.fillStyle = 'red';
      // detectionCtx.font = '16px Arial';
      // detectionCtx.fillText(label, x, y + 12);

      // Extract the detected region
      const extractedCanvas = document.createElement('canvas');
      extractedCanvas.width = width;
      extractedCanvas.height = height;
      const extractedCtx = extractedCanvas.getContext('2d');
      // Draw the extracted region onto the new canvas
      extractedCtx.drawImage(canvasRef.current, x, y, width, height, 0, 0, width, height);

      // Convert the new canvas to an image URL
      const detectionImageUrl = extractedCanvas.toDataURL('image/jpeg');

      // Determine player's move based on detection label
      const playerMove = detection.label

      addGameResult(playerMove, detectionImageUrl);
    }
  };

  const resetCapture = () => {
    setImage(null);
    setObjectDetected(false);
    previousImageDataRef.current = null;
    stableFrameCountRef.current = 0;
    setIsPaused(false);
    setVideoZIndex(3); // Bring video to front
    // startCaptureInterval();
    console.log('resetCapture called: isPaused set to false');
  };

  const resumeCapture = () => {
    resetCapture()
    // setIsPaused(false);
    // setObjectDetected(false);
    // setImage(null);  // Clear the detected image
    // setVideoZIndex(3); // Bring video to front
    // startCaptureInterval();
    // console.log('resumeCapture called: isPaused set to false');
  };

  return (
    <div className="App">
      <a href="https://research-triangle.ai" className="logo-container">
        <img src="images/logo.svg" alt="Research Triangle AI Society Logo" />
        <span className="org-name">Research Triangle AI Society</span>
      </a>
      {loading && (
        <Loader>
          {loading.progress
            ? `${loading.text} - ${loading.progress}%`
            : loading.text}
        </Loader>
      )}
      {/* image && */(
        <div className="inference-time-container" style={{display:"none"}}>
          <code className="code" ref={timeRef}></code>
        </div>
      )}
      <div className="header">
        <h1>Rock Paper Scissors: You vs The Machine</h1>
      </div>

      <div className="content">
        <video
          ref={videoRef}
          autoPlay
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            position: 'relative',
            transform: 'scaleX(-1)', // Add this line to mirror the video
            zIndex: videoZIndex, // Use the state variable
          }}
        />
        <img
          ref={imageRef}
          src={image || '#'}
          alt=""
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            objectFit: 'contain',
            visibility: image ? 'visible' : 'hidden',
            pointerEvents: 'none',
            zIndex: 1, // Position img above canvas and video
          }}
          onLoad={() => {
            // console.log("Image loaded");
            if (image && canvasRef.current && session) {
              // console.log("Calling detectImage with:", {
              //   image,
              //   canvas: canvasRef.current,
              //   session,
              //   scoreThreshold,
              //   modelInputShape,
              // });
              detectImage(
                image,
                canvasRef.current,
                session,
                scoreThreshold,
                modelInputShape,
                timeRef.current,
                handleDetectionResult
              );
            } else {
              // console.log("Conditions not met for detectImage:", {
              //   image,
              //   canvas: canvasRef.current,
              //   session,
              // });
            }
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none', // Allow interactions to pass through
            zIndex: 2,             // Position canvas above video but below other elements
          }}
        />
      </div>

      <div className="btn-container">
        {!isWebcamRunning && (
          <span>Dare challenge me? click <button onClick={startWebcam}>Start</button>!</span>
        )}
        {objectDetected && (
          <span>Click <button onClick={resumeCapture}>
            Continue
          </button> or press Space to play</span>
        )}
      </div>

      {games.length > 0 && (
        <div className="game-results">
          <p>Games played: {games.length}</p>
          <div className="winning-rate-container">
            <span>Your winning rate: <span style={{fontWeight: "bolder", color: "#4CAF50"}}>{calculateWinningRate().toFixed(2)}%</span></span>
            <div className="winning-bar-container">
              <div
                className="winning-bar"
                style={{
                  width: `${calculateWinningRate()}%`,
                }}
              ></div>
            </div>
          </div>
          <div className="game-moves">
            <div className="header-row">
              <div className="header">
                <span>You</span>
              </div>
              <div className="header">
                <span>The Machine</span>
              </div>
            </div>
            {games.slice().reverse().map(game => {
              const playerWon = (game.player === 0 && game.computer === 2) ||
                                (game.player === 1 && game.computer === 0) ||
                                (game.player === 2 && game.computer === 1);
              const computerWon = (game.computer === 0 && game.player === 2) ||
                                  (game.computer === 1 && game.player === 0) ||
                                  (game.computer === 2 && game.player === 1);
              const isDraw = game.player === game.computer;

              return (
                <div key={game.id} className="game-move">
                  <div className="player-move">
                    <img
                      src={`images/${game.player === 0 ? 'rock' : game.player === 1 ? 'paper' : 'scissors'}.png`}
                      alt={`Player Move ${game.id + 1}`}
                      style={{
                        filter: playerWon || isDraw ? 'none' : 'grayscale(100%) contrast(20%)',
                      }}
                    />
                  </div>
                  <div className="computer-move">
                    <img
                      src={`images/${game.computer === 0 ? 'rock2' : game.computer === 1 ? 'paper2' : 'scissors2'}.png`}
                      alt={`Computer Move ${game.id + 1}`}
                      style={{
                        filter: computerWon || isDraw ? 'none' : 'grayscale(100%) contrast(20%)',
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="footer-links">
        <a href="tech.html">Learn about the technology</a>
        <span className="dot-separator">•</span>
        <a href="https://github.com/research-triangle-ai/rock-paper-scissors">Source code on GitHub</a>
      </div>
    </div>
  );
};

export default App;
