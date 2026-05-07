/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { motion } from "motion/react";
import { 
  Camera, 
  Activity, 
  Target, 
  Terminal, 
  Cpu, 
  Github, 
  ChevronRight, 
  ArrowRight,
  Monitor,
  Database,
  Layers
} from "lucide-react";

export default function App() {
  return (
    <div className="min-h-screen bg-[#050505] text-white font-sans selection:bg-cyan-500/30">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 border-b border-white/5 bg-black/50 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <span className="font-semibold text-lg tracking-tight">Golf AI</span>
          </div>
          <div className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-sm font-medium text-gray-400 hover:text-white transition-colors">Features</a>
            <a href="#tech" className="text-sm font-medium text-gray-400 hover:text-white transition-colors">Technology</a>
            <a href="#install" className="text-sm font-medium text-gray-400 hover:text-white transition-colors">Installation</a>
          </div>
          <a 
            href="https://github.com/Asfand6417/Multiview-Golf-Swing-Analysis-and-Correction-using-Deep-Learning-Base-Post-Estimation" 
            target="_blank" 
            rel="no-referrer"
            className="flex items-center gap-2 bg-white text-black px-4 py-2 rounded-full text-sm font-semibold hover:bg-gray-200 transition-colors"
          >
            <Github size={18} />
            <span>Star on GitHub</span>
          </a>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 overflow-hidden">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid lg:grid-template-columns-[1.2fr_1fr] gap-12 items-center">
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
            >
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-bold uppercase tracking-wider mb-6">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                Deep Learning Powered Analysis
              </div>
              <h1 className="text-6xl md:text-8xl font-bold leading-[0.9] tracking-tighter mb-8 group">
                REDEFINE YOUR <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600">SWING MECHANICS</span>
              </h1>
              <p className="text-lg text-gray-400 max-w-xl mb-10 leading-relaxed">
                A state-of-the-art multiview pose estimation system that analyzes your golf swing with surgical precision using deep neural networks.
              </p>
              <div className="flex flex-wrap gap-4">
                <button className="px-8 py-4 bg-cyan-500 rounded-full font-bold hover:bg-cyan-400 transition-all transform hover:scale-105 flex items-center gap-2 shadow-lg shadow-cyan-500/20">
                  View Components <ArrowRight size={20} />
                </button>
                <div className="px-8 py-4 border border-white/10 rounded-full font-bold hover:bg-white/5 transition-all cursor-pointer">
                  Explore Models
                </div>
              </div>
            </motion.div>

            <motion.div 
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1, delay: 0.2 }}
              className="relative aspect-square rounded-3xl overflow-hidden border border-white/10 bg-gradient-to-br from-gray-900 to-black"
            >
              {/* Abstract Visual Representation */}
              <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1593111774240-d529f12cf4bb?q=80&w=2676&auto=format&fit=crop')] bg-cover bg-center opacity-40 mix-blend-luminosity" />
              <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent" />
              
              {/* Data Overlay Elements */}
              <div className="absolute top-6 left-6 p-4 rounded-xl bg-black/60 backdrop-blur-md border border-white/10">
                <div className="text-[10px] uppercase font-bold text-cyan-400 mb-1">Live Tracking</div>
                <div className="text-2xl font-mono">94.2% ACC</div>
              </div>
              <div className="absolute bottom-6 right-6 p-4 rounded-xl bg-black/60 backdrop-blur-md border border-white/10 max-w-[200px]">
                <div className="text-[10px] uppercase font-bold text-blue-400 mb-1">Phase Index</div>
                <div className="text-sm">Downswing Impact Transition Detected</div>
              </div>
              
              {/* Decorative Lines */}
              <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-50" viewBox="0 0 400 400">
                <motion.path 
                  d="M 50 350 L 150 150 L 250 200 L 350 50" 
                  fill="none" 
                  stroke="#22d3ee" 
                  strokeWidth="2"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 2, repeat: Infinity, repeatType: "reverse" }}
                />
                <circle cx="150" cy="150" r="4" fill="#22d3ee" />
                <circle cx="250" cy="200" r="4" fill="#22d3ee" />
              </svg>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Highlights Grid */}
      <section id="features" className="py-24 bg-[#080808] border-y border-white/5">
        <div className="max-width-7xl mx-auto px-6">
          <div className="mb-16">
            <h2 className="text-xs uppercase font-bold tracking-[0.2em] text-cyan-500 mb-3">Core Pillars</h2>
            <p className="text-4xl font-bold tracking-tight">Engineered for Accuracy</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: <Monitor className="text-cyan-400" />,
                title: "Multiview Sync",
                desc: "Simultaneous processing of front, side, and rear angles for a complete volumetric understanding of the swing."
              },
              {
                icon: <Target className="text-blue-400" />,
                title: "Joint-Level Precision",
                desc: "33+ body keypoints tracked via optimized MediaPipe layers with sub-pixel alignment."
              },
              {
                icon: <Cpu className="text-purple-400" />,
                title: "Phase Detection",
                desc: "Deep LSTM based sequence models that automatically index every swing phase from address to finish."
              }
            ].map((feature, i) => (
              <motion.div 
                key={i}
                whileHover={{ y: -5 }}
                className="p-8 rounded-2xl bg-white/[0.02] border border-white/5 hover:border-white/10 transition-all group"
              >
                <div className="w-12 h-12 rounded-xl bg-white/[0.05] flex items-center justify-center mb-6 border border-white/5 group-hover:border-cyan-500/50 transition-colors">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
                <p className="text-gray-400 text-sm leading-relaxed">{feature.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section id="tech" className="py-24">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row gap-12 items-center">
            <div className="flex-1">
              <h2 className="text-3xl font-bold mb-6 italic font-serif tracking-tight">The Neural Architecture</h2>
              <p className="text-gray-400 mb-8 max-w-lg">
                Built on a foundation of modern ML frameworks, the system utilizes custom CNN-LSTM architectures optimized for time-series pose data. 
                Spatial features from multiple viewpoints are fused to ensure robustness against occlusion.
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-xl border border-white/5 bg-white/[0.01]">
                  <div className="font-mono text-cyan-400 text-sm mb-1">Framework</div>
                  <div className="font-bold">PyTorch / TensorFlow</div>
                </div>
                <div className="p-4 rounded-xl border border-white/5 bg-white/[0.01]">
                  <div className="font-mono text-blue-400 text-sm mb-1">Vision</div>
                  <div className="font-bold">OpenCV / MediaPipe</div>
                </div>
              </div>
            </div>
            <div className="flex-1 grid grid-cols-2 gap-4 w-full">
              {[
                { icon: <Layers />, label: "Stacked Encoders" },
                { icon: <Database />, label: "Dataset Pipeline" },
                { icon: <Activity />, label: "Real-time Inference" },
                { icon: <Terminal />, label: "CLI Controller" }
              ].map((item, i) => (
                <div key={i} className="aspect-square rounded-2xl border border-white/5 bg-white/[0.01] flex flex-col items-center justify-center gap-4 hover:bg-white/[0.03] transition-colors">
                  <div className="text-gray-500">{item.icon}</div>
                  <span className="text-xs font-bold uppercase tracking-wider text-gray-400">{item.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Installation / Quick Start */}
      <section id="install" className="py-24 bg-white/5">
        <div className="max-w-7xl mx-auto px-6">
          <div className="max-w-3xl mx-auto text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Quick Integration</h2>
            <p className="text-gray-400">Get the analysis engine running in your local environment in minutes.</p>
          </div>
          
          <div className="max-w-2xl mx-auto rounded-2xl overflow-hidden border border-white/10 bg-black shadow-2xl">
            <div className="flex items-center gap-2 px-4 py-3 bg-white/5 border-b border-white/10">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <div className="w-3 h-3 rounded-full bg-yellow-500" />
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span className="text-xs text-gray-500 font-mono ml-4">asfand6417@terminal</span>
            </div>
            <div className="p-8 font-mono text-sm line-height-relaxed overflow-x-auto">
              <div className="flex gap-4">
                <span className="text-gray-600 select-none">1</span>
                <span><span className="text-cyan-400">git clone</span> https://github.com/Asfand6417/Golf-Swing-Analysis</span>
              </div>
              <div className="flex gap-4 opacity-50">
                <span className="text-gray-600 select-none">2</span>
                <span className="text-gray-400">cd Golf-Swing-Analysis</span>
              </div>
              <div className="flex gap-4">
                <span className="text-gray-600 select-none">3</span>
                <span><span className="text-blue-400">pip install</span> -r requirements.txt</span>
              </div>
              <div className="flex gap-4">
                <span className="text-gray-600 select-none">4</span>
                <span><span className="text-green-400">python</span> analyze.py --input video.mp4</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-20 border-t border-white/5 bg-black">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center gap-12">
            <div>
              <div className="flex items-center gap-2 mb-6">
                <Activity className="w-6 h-6 text-cyan-400" />
                <span className="font-bold text-xl tracking-tight">Golf AI</span>
              </div>
              <p className="text-gray-500 text-sm max-w-xs">
                Empowering athletes and coaches with deep learning precision.
              </p>
            </div>
            <div className="flex gap-12">
              <div className="flex flex-col gap-4">
                <span className="text-xs font-bold uppercase text-gray-400 tracking-widest">Project</span>
                <a href="#" className="text-sm text-gray-500 hover:text-white transition-colors">Documentation</a>
                <a href="#" className="text-sm text-gray-500 hover:text-white transition-colors">Demo</a>
                <a href="#" className="text-sm text-gray-500 hover:text-white transition-colors">Datasets</a>
              </div>
              <div className="flex flex-col gap-4">
                <span className="text-xs font-bold uppercase text-gray-400 tracking-widest">Connect</span>
                <a href="#" className="text-sm text-gray-500 hover:text-white transition-colors">GitHub</a>
                <a href="#" className="text-sm text-gray-500 hover:text-white transition-colors">Twitter</a>
                <a href="#" className="text-sm text-gray-500 hover:text-white transition-colors">LinkedIn</a>
              </div>
            </div>
          </div>
          <div className="mt-20 pt-8 border-t border-white/5 flex justify-between items-center text-xs text-gray-600">
            <span>© 2026 Multiview Golf Swing Analysis Project.</span>
            <span>Built by Asfand6417</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
