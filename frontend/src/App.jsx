import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Upload, Leaf, CheckCircle, Activity, AlertCircle, RefreshCw, Loader2, Sprout, ShieldCheck, Microscope } from 'lucide-react';

// Konfigurasi Model dan Label
const MODEL_PATH = '/model/model.json'; 
const EXPLAIN_API_URL = 'http://localhost:4000/explain';

const NAMA_KELAS = [
  'Bacterial Leaf Blight',
  'Brown Spot',
  'Healthy Rice Leaf',
  'Leaf Blast',
  'Leaf scald',
  'Narrow Brown Leaf Spot',
  'Neck_Blast',
  'Rice Hispa',
  'Sheath Blight'
];

export default function App() {
  // State Management
  const [model, setModel] = useState(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelError, setModelError] = useState(null);

  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null); 
  
  const imageRef = useRef(null);
  const resultsRef = useRef(null);

  // Load model
  useEffect(() => {
    async function loadModel() {
      try {
        const loadedModel = await tf.loadGraphModel(MODEL_PATH);
        setModel(loadedModel);
        setIsModelLoading(false);
      } catch (err) {
        console.error('Gagal memuat model:', err);
        setModelError('Gagal memuat model. Pastikan file model.json ada di folder public/model/');
        setIsModelLoading(false);
      }
    }
    loadModel();
  }, []);

  // --- 2. HANDLERS ---
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null); 
    } else {
      alert('Mohon pilih file gambar.');
    }
  };

  const analyzeImage = async () => {
    if (!model || !imageRef.current) return;

    setIsAnalyzing(true);
    setTimeout(() => {
      resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    try {
      // TensorFlow Process
      const tensor = tf.browser.fromPixels(imageRef.current)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();

      const predictions = model.execute(tensor);
      const data = await predictions.data();
      const maxIndex = predictions.argMax(1).dataSync()[0];
      
      const label = NAMA_KELAS[maxIndex];
      const confidence = (data[maxIndex] * 100).toFixed(2);

      tensor.dispose();
      predictions.dispose();

      // Backend Fetch
      let explanation = "Tidak ada penjelasan tambahan.";
      try {
        const response = await fetch(EXPLAIN_API_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label, confidence: Number(confidence) })
        });
        
        if (response.ok) {
          const resJson = await response.json();
          explanation = resJson.explanation || explanation;
        }
      } catch (backendErr) {
        console.warn("Backend error:", backendErr);
        explanation = "Gagal terhubung ke server AI Assistant (localhost:4000).";
      }

      setResult({ label, confidence, explanation });

    } catch (err) {
      console.error("Error prediksi:", err);
      alert("Terjadi kesalahan saat memproses gambar.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setResult(null);
  };

  // --- UI HELPERS ---
  const getConfidenceColor = (label) => {
    if (label.toLowerCase().includes('healthy')) return 'bg-emerald-600';
    return 'bg-amber-600';
  };

  const getStatusBadge = (label) => {
    if (label.toLowerCase().includes('healthy')) {
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-100 text-emerald-800 text-sm font-semibold border border-emerald-200">
          <CheckCircle className="w-4 h-4" /> Sehat
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-100 text-amber-800 text-sm font-semibold border border-amber-200">
        <AlertCircle className="w-4 h-4" /> Terinfeksi
      </span>
    );
  };

  return (
    // Gunakan bg-stone-50 untuk nuansa 'earthy'
    <div className="min-h-screen font-sans bg-stone-50 text-stone-800">
      
      {/* --- NAVBAR --- */}
      <nav className="sticky top-0 z-50 text-white shadow-lg bg-emerald-950">
        <div className="px-4 mx-auto max-w-7xl sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-white/10 backdrop-blur-sm">
                <Sprout className="w-6 h-6 text-lime-400" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-wide text-white">AI AGRI LAB</h1>
                <p className="text-[10px] text-emerald-200 uppercase tracking-widest">Smart Farming Solution</p>
              </div>
            </div>
            <div className="items-center hidden space-x-8 text-sm font-medium md:flex text-emerald-100">
              <a href="#hero" className="transition-colors hover:text-white">Beranda</a>
              <a href="#predictor" className="transition-colors hover:text-white">Analisis</a>
              <a href="#features" className="transition-colors hover:text-white">Teknologi</a>
            </div>
          </div>
        </div>
      </nav>

      {/* --- HERO SECTION --- */}
      <section id="hero" className="relative h-[500px] flex items-center justify-center overflow-hidden">
        {/* Background Image dengan Overlay Gradient Hijau Tua */}
        <div className="absolute inset-0 z-0">
          <img 
            src="/model/foto/sawah.jpg"
            alt="Sawah Padi" 
            className="object-cover w-full h-full"
          />
          <div className="absolute inset-0 bg-gradient-to-r from-emerald-950/90 via-emerald-900/80 to-transparent"></div>
        </div>

        <div className="relative z-10 w-full px-4 mx-auto max-w-7xl sm:px-6 lg:px-8">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 mb-6 text-xs font-bold tracking-wider uppercase border rounded-full bg-lime-500/20 border-lime-400/30 text-lime-300 backdrop-blur-md">
              <Microscope className="w-4 h-4" />
              Teknologi AI Untuk Pertanian
            </div>
            <h1 className="mb-6 text-4xl font-extrabold leading-tight text-white md:text-6xl">
              Smart Rice <br/>
              <span className="text-lime-400">Disease Detection</span>
            </h1>
            <p className="max-w-lg mb-8 text-lg leading-relaxed text-emerald-100">
              Deteksi dini penyakit tanaman padi menggunakan kecerdasan buatan. 
              Akurat, cepat, dan dirancang untuk ketahanan pangan Indonesia.
            </p>
            <a href="#predictor" className="inline-flex items-center justify-center px-8 py-4 text-base font-bold text-emerald-950 transition-all duration-200 bg-lime-400 rounded-lg hover:bg-lime-300 hover:shadow-[0_0_20px_rgba(163,230,53,0.5)]">
              Mulai Deteksi Sekarang
            </a>
          </div>
        </div>
      </section>

      <main className="relative z-20 px-4 py-12 mx-auto -mt-20 max-w-7xl sm:px-6 lg:px-8">
        
        {/* --- PREDICTOR CARD --- */}
        <section id="predictor" className="mb-20">
          <div className="overflow-hidden bg-white border shadow-xl rounded-2xl border-stone-200">
            
            {/* Header Card */}
            <div className="p-6 text-center border-b bg-stone-50 border-stone-200 md:p-8">
              <h2 className="text-2xl font-bold text-emerald-950">Laboratorium Digital</h2>
              <p className="mt-2 text-stone-500">Unggah sampel gambar daun untuk dianalisis oleh sistem.</p>
            </div>

            <div className="p-8 bg-white md:p-12">
              
              {/* Error/Loading State UI */}
              {isModelLoading && (
                <div className="flex items-center justify-center p-4 mb-8 border rounded-lg bg-amber-50 text-amber-800 border-amber-200">
                  <Loader2 className="w-5 h-5 mr-3 animate-spin" />
                  <span className="font-medium">Sedang menyiapkan model kecerdasan buatan...</span>
                </div>
              )}
              {modelError && (
                <div className="flex items-center justify-center p-4 mb-8 text-red-800 border border-red-200 rounded-lg bg-red-50">
                  <AlertCircle className="w-5 h-5 mr-3" />
                  <span className="font-medium">{modelError}</span>
                </div>
              )}

              <div className="grid items-start grid-cols-1 gap-12 lg:grid-cols-2">
                
                {/* --- INPUT AREA --- */}
                <div className="space-y-6">
                  <div className={`
                    relative group border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300
                    ${previewUrl ? 'border-emerald-500 bg-emerald-50/30' : 'border-stone-300 hover:border-emerald-400 hover:bg-stone-50'}
                    ${(!model || isModelLoading) ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}
                  `}>
                    <input 
                      type="file" 
                      className="absolute inset-0 z-10 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                      onChange={handleFileChange}
                      disabled={!model || isModelLoading}
                      accept="image/*"
                    />
                    
                    <div className="space-y-4 pointer-events-none">
                      <div className="flex items-center justify-center w-16 h-16 mx-auto transition-transform bg-white border rounded-full shadow-sm border-stone-100 group-hover:scale-110">
                        {previewUrl ? <RefreshCw className="w-8 h-8 text-emerald-600" /> : <Upload className="w-8 h-8 text-stone-400" />}
                      </div>
                      <div>
                        <p className="text-lg font-bold text-stone-700">
                          {previewUrl ? 'Ganti Sampel Gambar' : 'Unggah Foto Daun'}
                        </p>
                        <p className="mt-1 text-sm text-stone-500">Format: JPG, PNG (Maks 5MB)</p>
                      </div>
                    </div>
                  </div>

                  {previewUrl && !result && (
                    <button
                      onClick={analyzeImage}
                      disabled={isAnalyzing}
                      className="flex items-center justify-center w-full gap-3 py-4 font-bold text-white transition-all shadow-lg bg-emerald-700 hover:bg-emerald-800 rounded-xl shadow-emerald-900/20 disabled:opacity-70"
                    >
                      {isAnalyzing ? <Loader2 className="animate-spin" /> : <Microscope />}
                      {isAnalyzing ? 'Sedang Menganalisis...' : 'Jalankan Analisis'}
                    </button>
                  )}
                </div>

                {/* --- PREVIEW & RESULT AREA --- */}
                <div className="relative min-h-[300px] bg-stone-100 rounded-2xl border border-stone-200 overflow-hidden flex flex-col">
                  {/* Image Preview */}
                  {previewUrl ? (
                    <div className="relative w-full h-64 bg-black">
                      <img 
                        ref={imageRef} 
                        src={previewUrl} 
                        alt="Preview" 
                        className="object-contain w-full h-full"
                      />
                      {/* Scan Overlay Effect */}
                      {isAnalyzing && (
                        <div className="absolute inset-0 bg-emerald-500/20 animate-pulse border-b-4 border-emerald-400 w-full h-full origin-top transform transition-transform duration-[2s]"></div>
                      )}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center flex-1 p-8 text-stone-400">
                      <Leaf className="w-16 h-16 mb-4 opacity-20" />
                      <p>Pratinjau gambar akan muncul di sini</p>
                    </div>
                  )}

                  {/* Result Panel (Slide Up) */}
                  {result && (
                    <div ref={resultsRef} className="p-6 duration-500 bg-white border-t border-stone-200 animate-in slide-in-from-bottom-10 fade-in">
                      
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <p className="mb-1 text-xs font-bold tracking-wider uppercase text-stone-400">Hasil Identifikasi</p>
                          <h3 className="text-2xl font-bold leading-none text-emerald-950">{result.label}</h3>
                        </div>
                        {getStatusBadge(result.label)}
                      </div>

                      {/* Confidence Meter */}
                      <div className="mb-6">
                        <div className="flex justify-between mb-1 text-sm">
                          <span className="text-stone-500">Tingkat Akurasi AI</span>
                          <span className="font-bold text-emerald-700">{result.confidence}%</span>
                        </div>
                        <div className="w-full h-2 overflow-hidden rounded-full bg-stone-100">
                          <div 
                            className={`h-full rounded-full ${getConfidenceColor(result.label)} transition-all duration-1000`}
                            style={{ width: `${result.confidence}%` }}
                          ></div>
                        </div>
                      </div>

                      {/* Explanation Box */}
                      <div className="p-4 mb-6 border border-blue-100 rounded-lg bg-blue-50/50">
                        <div className="flex gap-2 mb-2 text-sm font-semibold text-blue-800">
                          <Activity className="w-4 h-4" /> Analisis Pakar AI
                        </div>
                        <div className="space-y-2 text-sm leading-relaxed text-stone-700">
                           {result.explanation.split('\n').map((line, i) => (
                             <p key={i}>{line}</p>
                           ))}
                        </div>
                      </div>

                      <button 
                        onClick={handleReset}
                        className="w-full py-3 text-sm font-semibold transition-colors rounded-lg bg-stone-100 hover:bg-stone-200 text-stone-600"
                      >
                        Analisis Sampel Lain
                      </button>
                    </div>
                  )}
                </div>

              </div>
            </div>
          </div>
        </section>

        {/* --- FEATURES GRID --- */}
        <section id="features" className="pb-12">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-bold text-emerald-950">Keunggulan Sistem</h2>
            <p className="mt-2 text-stone-500">Menggabungkan agrikultur tradisional dengan teknologi masa depan.</p>
          </div>
          
          <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
            <FeatureCard 
              icon={<ShieldCheck className="w-8 h-8 text-emerald-600" />}
              title="Akurasi Tinggi"
              desc="Menggunakan model Computer Vision yang dilatih pada ribuan dataset penyakit tanaman padi global."
            />
            <FeatureCard 
              icon={<Activity className="w-8 h-8 text-emerald-600" />}
              title="Real-Time Processing"
              desc="Pemrosesan gambar dilakukan langsung menggunakan Artificial Intelligence."
            />
            <FeatureCard 
              icon={<Sprout className="w-8 h-8 text-emerald-600" />}
              title="Solusi Berkelanjutan"
              desc="Membantu petani mengurangi penggunaan pestisida berlebih dengan identifikasi penyakit yang tepat sasaran."
            />
          </div>
        </section>

      </main>

      {/* --- FOOTER --- */}
      <footer className="py-12 border-t bg-emerald-950 text-emerald-400/60 border-emerald-900">
        <div className="px-4 mx-auto text-center max-w-7xl">
          <div className="flex items-center justify-center gap-2 mb-4">
             <Sprout className="w-6 h-6 text-emerald-600" />
             <span className="text-xl font-bold tracking-wide text-emerald-100">AI AGRI LAB</span>
          </div>
          <p className="mb-4 text-sm">Dikembangkan untuk masa depan pertanian Indonesia.</p>
          <p className="text-xs">&copy; 2026 AI AGRI LAB. All Rights Reserved.</p>
        </div>
      </footer>
    </div>
  );
}

// Component Kartu Fitur
function FeatureCard({ icon, title, desc }) {
  return (
    <div className="p-8 transition-all duration-300 bg-white border shadow-sm rounded-2xl border-stone-100 hover:shadow-lg hover:-translate-y-1">
      <div className="flex items-center justify-center mb-6 w-14 h-14 bg-emerald-50 rounded-xl">
        {icon}
      </div>
      <h3 className="mb-3 text-xl font-bold text-stone-800">{title}</h3>
      <p className="text-sm leading-relaxed text-stone-500">{desc}</p>
    </div>
  );
}