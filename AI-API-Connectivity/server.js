require("dotenv").config();

const express = require("express");
const path = require("path");
const { GoogleGenAI } = require("@google/genai");

const app = express();
const port = 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const apiKey = process.env.GEMINI_API_KEY;

const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;

// Display the GUI
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// Receive prompt and return AI response
app.post("/chat", async (req, res) => {
  try {
    const prompt = req.body.prompt;

    if (typeof prompt !== "string" || !prompt.trim()) {
      return res.status(400).json({
        error: "Please enter a prompt."
      });
    }

    if (!ai) {
      return res.status(500).json({
        error: "API key is missing. Check your .env file."
      });
    }

    const result = await ai.models.generateContent({
  model: "gemini-3.8-flash",
  contents: `Answer clearly and concisely. Use simple language and a maximum of 150 words.

Question: ${prompt.trim()}`,
  config: {
    maxOutputTokens: 256,
    temperature: 0.5
  }
});

    return res.json({
      response: result.text || "No response was returned."
    });

  } catch (error) {
    console.error("Gemini API error:", error);

    const message = String(error.message || "");

    if (
      message.includes('"code":503') ||
      message.includes("UNAVAILABLE") ||
      message.includes("high demand")
    ) {
      return res.status(503).json({
        error: "Gemini is temporarily busy. Please try again shortly."
      });
    }

    return res.status(500).json({
      error: "Gemini request failed. Check the server terminal for details."
    });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
