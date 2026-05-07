import React, { useState } from 'react';
import { motion } from "motion/react";
import { 
  Camera, Activity, Target, MessageSquare, 
  Settings, History, Play, Shield, Menu, X, ChevronRight, Search
} from "lucide-react";

export default function App() {
  const [activeTab, setActiveTab] = useState('analyze');

  return (
    <div className="flex h-screen bg-[#0a0a0a] text-white font-sans overflow-hidden">
      {/* Sidebar Navigation */}
      <nav className="w-20 lg:w-64 border-r border-white/5 flex flex-col items-center lg:items-start py-8 px-4 bg-black/40 backdrop-blur-xl">
        <div className="flex items-center gap-3 mb-12 px-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Activity className="text-white w-6 h-6" />
          </div>
          <span className="hidden lg:block font-bold text-xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">GOLF AI</span>
        </div>

        <div className="flex-1 w-full space-y-2">
          <SidebarItem icon={<Camera />} label="Analysis" active={activeTab === 'analyze'} onClick={() => setActiveTab('analyze')} />
          <SidebarItem icon={<MessageSquare />} label="Coach Insights" active={activeTab === 'chat'} onClick={() => setActiveTab('chat')} />
          <SidebarItem icon={<History />} label="Swing History" active={activeTab === 'history'} onClick={() => setActiveTab('history')} />
          <SidebarItem icon={<Target />} label="Performance" active={activeTab === 'goals'} onClick={() => setActiveTab('goals')} />
        </div>

        <div className="w-full pt-6 border-t border-white/5">
          <SidebarItem icon={<Settings />} label="Settings" />
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-20 border-b border-white/5 flex items-center justify-between px-8 bg-black/20 backdrop-blur-md">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-semibold">Active Session: Multiview 360°</h2>
            <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-[10px] font-bold">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" /> SYSTEM READY
            </span>
          </div>
          <div className="flex items-center gap-6">
            <div className="relative hidden md:block">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
              <input className="bg-white/5 border border-white/10 rounded-full py-2 pl-10 pr-4 text-xs w-64 focus:outline-none focus:border-cyan-500/50" placeholder="Search sessions..." />
            </div>
            <button className="bg-cyan-500 hover:bg-cyan-400 text-black text-xs font-bold px-5 py-2.5 rounded-full transition-all">
              New Analysis
            </button>
          </div>
        </header>

        {/* Workspace Layout */}
        <div className="flex-1 flex flex-col lg:flex-row p-6 gap-6 overflow-hidden">
          {/* Analysis Viewport */}
          <div className="flex-[1.8] flex flex-col gap-6 overflow-hidden">
             <div className="flex-1 bg-black rounded-3xl border border-white/10 relative overflow-hidden group shadow-2xl">
                <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-transparent pointer-events-none" />
                <div className="absolute inset-0 flex items-center justify-center">
                   <div className="text-center group-hover:scale-105 transition-transform duration-500">
                      <div className="w-20 h-20 rounded-full bg-white/5 backdrop-blur-xl border border-white/10 flex items-center justify-center mx-auto mb-6 shadow-2xl">
                        <Play className="w-8 h-8 text-cyan-400 fill-cyan-400" />
                      </div>
                      <p className="text-gray-400 font-medium tracking-wide">Drop video file or click to stream</p>
                      <p className="text-gray-600 text-xs mt-2 font-mono">SUPPORTED: MP4, MOV, WEBM (60FPS PREFERRED)</p>
                   </div>
                </div>
                
                {/* Visual Overlays */}
                <div className="absolute top-6 left-6 flex gap-3">
                  <Badge label="FRONT" active />
                  <Badge label="SIDE" />
                  <Badge label="REAR" />
                </div>
             </div>

             {/* Metrics Strip */}
             <div className="h-32 grid grid-cols-4 gap-4">
                <MetricCard label="Club Speed" value="108" unit="mph" />
                <MetricCard label="Attack Angle" value="-2.4°" unit="down" />
                <MetricCard label="Face Angle" value="0.8°" unit="open" />
                <MetricCard label="Smash Factor" value="1.48" unit="idx" />
             </div>
          </div>

          {/* AI Coach Insights Panel */}
          <aside className="flex-1 bg-white/[0.02] rounded-3xl border border-white/10 flex flex-col backdrop-blur-sm shadow-xl">
            <div className="p-6 border-b border-white/5 flex items-center justify-between">
              <h3 className="font-bold flex items-center gap-2 text-sm tracking-wider uppercase text-gray-400">
                <Shield className="w-4 h-4 text-cyan-400" />
                Coach Insights
              </h3>
              <span className="text-[10px] font-mono text-cyan-500/50">v4.2.0</span>
            </div>
            
            <div className="flex-1 p-6 overflow-y-auto space-y-6">
              <ChatMessage 
                role="coach" 
                text="Hello Asfand! I'm ready to analyze your mechanics. Upload your swing from multiple angles, and I'll provide a frame-by-frame breakdown of your kinetic chain." 
              />
              <div className="p-4 rounded-2xl bg-cyan-500/5 border border-cyan-500/10">
                <div className="text-[10px] font-bold text-cyan-400 mb-2 uppercase">Pro Tip</div>
                <p className="text-sm text-gray-300 leading-relaxed italic">
                  "Focus on your lead hip rotation during the transition phase. You're currently seeing a 12% loss in power due to early extension."
                </p>
              </div>
            </div>

            <div className="p-6">
              <div className="relative group">
                <input 
                  className="w-full bg-white/5 border border-white/10 rounded-2xl py-4 pl-6 pr-14 text-sm focus:outline-none focus:border-cyan-500/50 transition-all"
                  placeholder="Ask about your downswing..."
                />
                <button className="absolute right-3 top-1/2 -translate-y-1/2 w-10 h-10 bg-cyan-500 rounded-xl flex items-center justify-center hover:bg-cyan-400 transition-colors shadow-lg shadow-cyan-500/20">
                  <ChevronRight className="text-black w-5 h-5" />
                </button>
              </div>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}

// Sub-components
function SidebarItem({ icon, label, active = false, onClick }: any) {
  return (
    <button onClick={onClick} className={`w-full flex items-center gap-4 p-4 rounded-2xl transition-all ${active ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 shadow-inner' : 'text-gray-500 hover:text-white hover:bg-white/5'}`}>
      {React.cloneElement(icon, { size: 20 })}
      <span className="hidden lg:block font-semibold text-sm">{label}</span>
    </button>
  );
}

function Badge({ label, active = false }: any) {
  return (
    <div className={`px-3 py-1 rounded-lg text-[10px] font-bold border transition-colors ${active ? 'bg-cyan-500 border-cyan-500 text-black shadow-lg shadow-cyan-500/30' : 'bg-black/60 border-white/10 text-gray-400'}`}>
      {label}
    </div>
  );
}

function MetricCard({ label, value, unit }: any) {
  return (
    <div className="bg-white/[0.03] border border-white/5 rounded-2xl p-4 flex flex-col justify-center">
      <div className="text-[10px] text-gray-500 font-bold uppercase mb-1">{label}</div>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-mono font-bold text-white">{value}</span>
        <span className="text-[10px] text-cyan-500 font-medium">{unit}</span>
      </div>
    </div>
  );
}

function ChatMessage({ role, text }: any) {
  return (
    <div className={`flex gap-3 ${role === 'coach' ? '' : 'flex-row-reverse'}`}>
      <div className={`w-8 h-8 rounded-lg flex-shrink-0 flex items-center justify-center ${role === 'coach' ? 'bg-cyan-500/20 text-cyan-400' : 'bg-white/10 text-white'}`}>
        {role === 'coach' ? <Shield size={16} /> : <div className="text-[10px] font-bold">YOU</div>}
      </div>
      <div className={`p-4 rounded-2xl text-xs leading-relaxed ${role === 'coach' ? 'bg-white/[0.05] border border-white/5 text-gray-300' : 'bg-cyan-500 text-black font-medium'}`}>
        {text}
      </div>
    </div>
  );
}