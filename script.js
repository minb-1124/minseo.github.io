const canvas = document.querySelector("#pathCanvas");
const ctx = canvas.getContext("2d");

const points = [
  { x: 0.11, y: 0.72, value: 0.18, label: "scan" },
  { x: 0.24, y: 0.52, value: 0.31, label: "enumerate" },
  { x: 0.39, y: 0.61, value: 0.45, label: "foothold" },
  { x: 0.53, y: 0.38, value: 0.62, label: "pivot" },
  { x: 0.69, y: 0.45, value: 0.74, label: "privilege" },
  { x: 0.84, y: 0.25, value: 0.91, label: "goal" },
];

function fitCanvas() {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * scale);
  canvas.height = Math.round(rect.height * scale);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
}

function drawGrid(width, height) {
  ctx.strokeStyle = "rgba(239, 240, 232, 0.08)";
  ctx.lineWidth = 1;

  for (let x = 36; x < width; x += 58) {
    ctx.beginPath();
    ctx.moveTo(x, 28);
    ctx.lineTo(x, height - 32);
    ctx.stroke();
  }

  for (let y = 34; y < height; y += 52) {
    ctx.beginPath();
    ctx.moveTo(30, y);
    ctx.lineTo(width - 30, y);
    ctx.stroke();
  }
}

function draw(timestamp = 0) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);

  drawGrid(width, height);

  const time = timestamp / 1000;
  const mapped = points.map((point) => ({
    ...point,
    px: point.x * width,
    py: point.y * height + Math.sin(time + point.value * 8) * 7,
  }));

  ctx.lineWidth = 4;
  ctx.strokeStyle = "#d7ebdd";
  ctx.beginPath();
  mapped.forEach((point, index) => {
    if (index === 0) {
      ctx.moveTo(point.px, point.py);
    } else {
      const prev = mapped[index - 1];
      const cx = (prev.px + point.px) / 2;
      ctx.bezierCurveTo(cx, prev.py, cx, point.py, point.px, point.py);
    }
  });
  ctx.stroke();

  mapped.forEach((point, index) => {
    const radius = 10 + point.value * 8;
    const pulse = Math.sin(time * 2 + index) * 3;

    ctx.fillStyle = index === mapped.length - 1 ? "#c38a22" : "#1f7a58";
    ctx.beginPath();
    ctx.arc(point.px, point.py, radius + pulse, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = "rgba(247, 246, 239, 0.75)";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#f7f6ef";
    ctx.font = "700 13px ui-sans-serif, system-ui";
    ctx.fillText(point.label, point.px + 14, point.py - 14);
  });

  const curveLeft = width * 0.1;
  const curveTop = height * 0.82;
  const curveWidth = width * 0.78;
  ctx.strokeStyle = "rgba(238, 240, 232, 0.28)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(curveLeft, curveTop);
  ctx.lineTo(curveLeft + curveWidth, curveTop);
  ctx.stroke();

  ctx.strokeStyle = "#b54843";
  ctx.lineWidth = 3;
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = curveLeft + (curveWidth / (points.length - 1)) * index;
    const y = curveTop - point.value * 98;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();

  requestAnimationFrame(draw);
}

fitCanvas();
draw();
window.addEventListener("resize", fitCanvas);
