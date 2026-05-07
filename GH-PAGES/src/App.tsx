/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { motion } from "motion/react";
import { 
  Camera, Activity, Target, MessageSquare, 
  Settings, History, Play, Shield, Menu, X 
} from "lucide-react";

export default function App() {
  const [activeTab, setActiveTab] = useState('analyze');

  return (
    <div className="flex h-screen bg-[#0a0a0a] text-white font-sans overflow-hidden">
      {/* Sidebar Navigation */}
      <nav className="w-20 lg:w-64 border-r border-white/5 flex flex-col items-center lg:items-start py-8 px-4 bg-black/20">
        <div className="flex items-center gap-3 mb-12 px-2">
          <div className="w-10 h-10 rounded-xl bg-cyan-500 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Activity className="text-white" />
          </div>
          <span className="hidden lg:block font-bold text-xl tracking-tight">GOLF AI</span>
        </div>

        <div className="flex-1 w-full space-y-2">
          <SidebarItem icon={<Camera />} label="Analyze" active={activeTab === 'analyze'} onClick={() => setActiveTab('analyze')} />
          <SidebarItem icon={<MessageSquare />} label="AI Coach" active={activeTab === 'chat'} onClick={() => setActiveTab('chat')} />
          <SidebarItem icon={<History />} label="History" active={activeTab === 'history'} onClick={() => setActiveTab('history')} />
          <SidebarItem icon={<Target />} label="Goals" active={activeTab === 'goals'} onClick={() => setActiveTab('goals')} />
        </div>

        <div className="w-full pt-6 border-t border-white/5">
          <SidebarItem icon={<Settings />} label="Settings" />
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col overflow-hidden bg-gradient-to-br from-[#0a0a0a] to-[#111]">
        {/* Header */}
        <header className="h-20 border-b border-white/5 flex items-center justify-between px-8 bg-black/40 backdrop-blur-md">
          <h2 className="text-xl font-semibold">Golf Swing Analysis</h2>
          <div className="flex items-center gap-4">
            <div className="px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-mono">
              GPU ACCELERATION: ON
            </div>
            <div className="w-10 h-10 rounded-full bg-white/5 border border-white/10 flex items-center justify-center">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            </div>
          </div>
        </header>

        {/* Workspace */}
        <div className="flex-1 flex flex-col lg:flex-row p-6 gap-6 overflow-hidden">
          {/* Video / Analysis Viewport */}
          <div className="flex-[1.5] bg-black rounded-3xl border border-white/10 relative overflow-hidden group">
            <div className="absolute inset-0 flex items-center justify-center bg-white/5">
               <div className="text-center">
                  <Play className="w-16 h-16 text-cyan-500 mx-auto mb-4 opacity-50 group-hover:opacity-100 transition-opacity" />
                  <p className="text-gray-500 font-medium">Upload or Stream Swing Video</p>
               </div>
            </div>
            {/* Overlay UI */}
            <div className="absolute bottom-6 left-6 right-6 flex justify-between items-end">
              <div className="p-4 bg-black/60 backdrop-blur-md border border-white/10 rounded-2xl">
                <div className="text-[10px] text-cyan-400 font-bold uppercase mb-1">Phase Tracking</div>
                <div className="text-sm font-mono">WAITING FOR INPUT...</div>
              </div>
            </div>
          </div>

          {/* AI Insights Panel */}
          <aside className="flex-1 bg-white/[0.02] rounded-3xl border border-white/10 flex flex-col backdrop-blur-sm">
            <div className="p-6 border-b border-white/5">
              <h3 className="font-bold flex items-center gap-2">
                <Shield className="w-4 h-4 text-cyan-400" />
                Coach Insights
              </h3>
            </div>
            <div className="flex-1 p-6 overflow-y-auto space-y-4">
              <div className="p-4 rounded-2xl bg-cyan-500/5 border border-cyan-500/10 text-sm leading-relaxed text-gray-300">
                Welcome, Asfand. Upload a video from the front or side view to begin your AI-powered mechanical breakdown.
              </div>
            </div>
            <div className="p-4">
              <div className="relative">
                <input 
                  className="w-full bg-white/5 border border-white/10 rounded-2xl py-4 px-6 text-sm focus:outline-none focus:border-cyan-500/50 transition-colors"
                  placeholder="Ask the AI Coach..."
                />
                <button className="absolute right-3 top-3 px-4 py-1.5 bg-cyan-500 rounded-xl text-xs font-bold hover:bg-cyan-400 transition-colors">
                  Send
                </button>
              </div>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}

function SidebarItem({ icon, label, active = false, onClick }: any) {
  return (
    <button 
      onClick={onClick}
      className={`w-full flex items-center gap-4 p-4 rounded-2xl transition-all ${
        active ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20' : 'text-gray-500 hover:text-white hover:bg-white/5'
      }`}
    >
      {React.cloneElement(icon, { size: 20 })}
      <span className="hidden lg:block font-medium text-sm">{label}</span>
    </button>
  );
}
