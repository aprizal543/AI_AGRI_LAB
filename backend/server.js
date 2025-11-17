import express from "express";
import fetch from "node-fetch";
import dotenv from "dotenv";
import cors from "cors";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 4000;
const GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent";


app.use(cors());
app.use(express.json());

app.post("/explain", async (req, res) => {
  const { label, confidence } = req.body;
  if (!process.env.GEMINI_API_KEY) {
    return res.status(500).json({ error: "API key missing" });
  }
  if (!label) {
    return res.status(400).json({ error: "label required" });
  }

  // Instruksi LLM fokus pada penyakit daun padi
  const prompt = `
Anda adalah asisten agronomi digital. Gunakan data berikut:
- Prediksi penyakit: ${label}
Balas dua kalimat: gejala singkat + saran cara penganannya maksimal 2 paragraf.
Jangan uraikan proses.

`.trim();

  try {
    let data = null;

    for (let attempt = 1; attempt <= 3; attempt++) {
      const response = await fetch(`${GEMINI_ENDPOINT}?key=${process.env.GEMINI_API_KEY}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ role: "user", parts: [{ text: prompt }] }],
          generationConfig: {
                            temperature: 0.4,
                            maxOutputTokens: 700
                            }
        })
      });

      if (response.status === 503 && attempt < 3) {
        console.warn(`Gemini 503, mencoba ulang (${attempt}/3)...`);
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
        continue;
      }

      if (!response.ok) {
        const errText = await response.text();
        console.error("Gemini API error", response.status, errText);
        return res.status(response.status).json({ error: "Gemini API error", detail: errText });
      }

      data = await response.json();
      break;
    }

    if (!data) {
      return res.status(503).json({ error: "Gemini tidak merespons setelah beberapa percobaan." });
    }

    console.log("Gemini raw:", JSON.stringify(data, null, 2));
    const parts = data?.candidates?.[0]?.content?.parts || [];
    const text = parts
        .map(part => part.text || part?.inlineData?.data || "")
        .join("\n")
        .trim();
    res.json({ explanation: text || "Tidak ada penjelasan." });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Gagal menghubungi Gemini" });
  }
});

app.listen(PORT, () => {
  console.log(`Backend berjalan di port ${PORT}`);
});
