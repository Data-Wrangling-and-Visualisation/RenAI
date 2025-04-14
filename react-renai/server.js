import express from "express";
import cors from "cors";
import { exec } from "child_process";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());

// Эндпоинт для embeddings
app.get("/api/embeddings", (req, res) => {
  const fileParam = req.query.file || "a-bashi-bazouk.pt";
  const category = req.query.category || "embeddings";
  // Вызываем Python‑скрипт для конвертации .pt файла в JSON
  const command = `python3 convert_embeddings.py ${category} ${encodeURIComponent(
    fileParam
  )}`;
  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error("exec error (embeddings):", error);
      return res.status(500).json({ error: "Ошибка конвертации embeddings" });
    }
    try {
      const jsonData = JSON.parse(stdout);
      res.json(jsonData);
    } catch (parseError) {
      console.error("Ошибка парсинга JSON (embeddings):", parseError);
      res.status(500).json({ error: "Ошибка парсинга данных embeddings" });
    }
  });
});

// Эндпоинт для GradCAM
app.get("/api/gradcam", (req, res) => {
  const fileParam = req.query.file || "default.pt";
  const category = req.query.category || "gradcam";
  const command = `python3 convert_gradcam.py ${category} ${encodeURIComponent(
    fileParam
  )}`;
  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error("exec error (gradcam):", error);
      return res.status(500).json({ error: "Ошибка конвертации gradcam" });
    }
    try {
      const jsonData = JSON.parse(stdout);
      res.json(jsonData);
    } catch (parseError) {
      console.error("Ошибка парсинга JSON (gradcam):", parseError);
      res.status(500).json({ error: "Ошибка парсинга данных gradcam" });
    }
  });
});

// Эндпоинт для Attention
app.get("/api/attention", (req, res) => {
  const fileParam = req.query.file || "default.pt";
  const category = req.query.category || "attention";
  const command = `python3 convert_attention.py ${category} ${encodeURIComponent(
    fileParam
  )}`;
  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error("exec error (attention):", error);
      return res.status(500).json({ error: "Ошибка конвертации attention" });
    }
    try {
      const jsonData = JSON.parse(stdout);
      res.json(jsonData);
    } catch (parseError) {
      console.error("Ошибка парсинга JSON (attention):", parseError);
      res.status(500).json({ error: "Ошибка парсинга данных attention" });
    }
  });
});

app.listen(PORT, () => {
  console.log(`Сервер запущен на http://localhost:${PORT}`);
});