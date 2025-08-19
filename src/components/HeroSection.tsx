import React from 'react';
import { Button } from '@/components/ui/button';
import { Sparkles, Search, Cpu, Zap, Shield, TrendingUp } from 'lucide-react';

interface HeroSectionProps {
  onStartChat: () => void;
}

export const HeroSection: React.FC<HeroSectionProps> = ({ onStartChat }) => {
  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-background via-background to-card">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-float"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '2s' }}></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-secondary/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '4s' }}></div>
      </div>

      <div className="relative z-10 text-center max-w-4xl mx-auto px-6">
        {/* Logo & Brand */}
        <div className="mb-8 animate-float">
          <div className="inline-block p-4 bg-gradient-to-r from-primary to-accent rounded-3xl shadow-2xl mb-6">
            <Cpu className="w-16 h-16 text-primary-foreground" />
          </div>
          <h1 className="text-6xl md:text-7xl font-bold mb-4">
            <span className="bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent animate-gradient">
              ZeeVice AI
            </span>
          </h1>
          <p className="text-xl md:text-2xl text-muted-foreground mb-12 max-w-2xl mx-auto leading-relaxed">
            Your intelligent shopping assistant for smartphones in Egypt. 
            Powered by advanced AI to find your perfect device.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="group p-6 bg-card/50 backdrop-blur-sm border border-border/50 rounded-2xl hover:border-accent/50 transition-all duration-300 hover:shadow-lg">
            <Search className="w-8 h-8 text-primary mb-4 group-hover:scale-110 transition-transform" />
            <h3 className="text-lg font-semibold mb-2 text-foreground">Hybrid Search</h3>
            <p className="text-sm text-muted-foreground">Combines graph databases and vector search for precise results</p>
          </div>
          
          <div className="group p-6 bg-card/50 backdrop-blur-sm border border-border/50 rounded-2xl hover:border-accent/50 transition-all duration-300 hover:shadow-lg">
            <Zap className="w-8 h-8 text-accent mb-4 group-hover:scale-110 transition-transform" />
            <h3 className="text-lg font-semibold mb-2 text-foreground">Real-time Data</h3>
            <p className="text-sm text-muted-foreground">Live pricing and availability from major Egyptian retailers</p>
          </div>
          
          <div className="group p-6 bg-card/50 backdrop-blur-sm border border-border/50 rounded-2xl hover:border-accent/50 transition-all duration-300 hover:shadow-lg">
            <Shield className="w-8 h-8 text-secondary mb-4 group-hover:scale-110 transition-transform" />
            <h3 className="text-lg font-semibold mb-2 text-foreground">Smart Recommendations</h3>
            <p className="text-sm text-muted-foreground">AI-powered suggestions based on your needs and budget</p>
          </div>
        </div>

        {/* CTA Section */}
        <div className="space-y-6">
          <Button 
            onClick={onStartChat}
            variant="hero" 
            size="lg"
            className="text-lg px-12 py-6 h-auto shadow-2xl hover:shadow-[0_0_50px_hsl(var(--primary)/0.5)]"
          >
            <Sparkles className="w-6 h-6 mr-2" />
            Start Shopping with AI
          </Button>
          
          <div className="flex items-center justify-center space-x-8 text-sm text-muted-foreground">
            <div className="flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-success" />
              <span>10K+ Products</span>
            </div>
            <div className="w-px h-4 bg-border"></div>
            <div className="flex items-center space-x-2">
              <Shield className="w-4 h-4 text-primary" />
              <span>Verified Stores</span>
            </div>
            <div className="w-px h-4 bg-border"></div>
            <div className="flex items-center space-x-2">
              <Zap className="w-4 h-4 text-accent" />
              <span>Real-time Prices</span>
            </div>
          </div>
        </div>

        {/* Demo Query Examples */}
        <div className="mt-16 p-8 bg-card/30 backdrop-blur-sm border border-border/30 rounded-3xl">
          <h3 className="text-lg font-semibold mb-6 text-foreground">Try asking me:</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-left">
            <div className="p-4 bg-background/50 rounded-xl border border-border/30">
              <p className="text-sm text-foreground">"Find me a gaming phone under 30,000 EGP"</p>
            </div>
            <div className="p-4 bg-background/50 rounded-xl border border-border/30">
              <p className="text-sm text-foreground">"Best iPhone with good camera for photography"</p>
            </div>
            <div className="p-4 bg-background/50 rounded-xl border border-border/30">
              <p className="text-sm text-foreground">"Samsung phones with 5G and long battery life"</p>
            </div>
            <div className="p-4 bg-background/50 rounded-xl border border-border/30">
              <p className="text-sm text-foreground">"Compare flagship phones from B.Tech and Noon"</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};