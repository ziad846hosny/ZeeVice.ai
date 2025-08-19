import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Send, Bot, User, Sparkles, Cpu, ExternalLink, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { api, type ProductCard } from '@/services/api';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  searchResults?: {
    search_results: ProductCard[];
    mentioned_product_ids?: string[];
  };
  error?: boolean;
}

const ProductCardComponent: React.FC<{ product: ProductCard }> = ({ product }) => (
  <Card className="p-4 bg-gradient-to-br from-card to-card/80 border-border/50 hover:border-accent/50 transition-all duration-300 hover:shadow-lg group">
    <div className="space-y-3">
      <div className="flex justify-between items-start">
        <h3 className="font-semibold text-foreground group-hover:text-accent transition-colors">
          {product.title}
        </h3>
        <span className="text-sm bg-accent/10 text-accent px-2 py-1 rounded-lg">
          {product.brand}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
        {product.display_size && <div>Display: {product.display_size}</div>}
        {product.ram && <div>RAM: {product.ram}</div>}
        {product.storage && <div>Storage: {product.storage}</div>}
        {product.battery && <div>Battery: {product.battery}</div>}
        {product.main_camera && <div>Camera: {product.main_camera}</div>}
        {product.network && <div>Network: {product.network}</div>}
      </div>

      {(product.display_type || product.cores || product.os) && (
        <div className="text-xs text-muted-foreground space-y-1">
          {product.display_type && <div>Display Type: {product.display_type}</div>}
          {product.cores && <div>Processor: {product.cores}</div>}
          {product.os && <div>OS: {product.os}</div>}
        </div>
      )}

      <div className="flex justify-between items-center pt-2 border-t border-border/30">
        <div className="text-right">
          <div className="text-2xl font-bold text-primary">{product.price.toLocaleString()} EGP</div>
          <div className="text-sm text-muted-foreground">{product.store}</div>
        </div>
        {product.url && (
          <Button
            variant="outline"
            size="sm"
            className="ml-auto"
            onClick={() => window.open(product.url, '_blank')}
          >
            <ExternalLink className="w-4 h-4" />
            View
          </Button>
        )}
      </div>
    </div>
  </Card>
);

const TypingIndicator: React.FC = () => (
  <div className="flex items-center space-x-2 p-4">
    <div className="flex space-x-1">
      <div className="w-2 h-2 bg-accent rounded-full animate-pulse"></div>
      <div className="w-2 h-2 bg-accent rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
      <div className="w-2 h-2 bg-accent rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
    </div>
    <span className="text-sm text-muted-foreground">ZeeVice AI is thinking...</span>
  </div>
);

export const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'Hello! I\'m ZeeVice AI, your smart shopping assistant for smartphones in Egypt. I can help you find the perfect phone based on your needs, budget, and preferences. What are you looking for today?',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = input;
    setInput('');
    setIsTyping(true);

    try {
      const chatResponse = await api.sendMessage(currentInput);

      // DEBUG: print all products to console
      console.log("=== Chat Response ===");
      console.log(chatResponse);
      console.log("Search results array:", chatResponse.search_results);
      console.log("Mentioned product IDs:", chatResponse.mentioned_product_ids);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: chatResponse.response,
        timestamp: new Date(),
        searchResults: {
          search_results: chatResponse.search_results || [],
          mentioned_product_ids: chatResponse.mentioned_product_ids
        }
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat failed:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I\'m having trouble connecting to the AI service right now. Please check your backend.',
        timestamp: new Date(),
        error: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto">
      {/* Header */}
      <div className="p-6 border-b border-border/50 bg-card/30 backdrop-blur-sm">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-r from-primary to-accent rounded-xl flex items-center justify-center">
            <Cpu className="w-5 h-5 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              ZeeVice AI
            </h1>
            <p className="text-sm text-muted-foreground">Smart Phone Search Assistant</p>
          </div>
          <div className="ml-auto flex items-center space-x-2">
            <div className="w-2 h-2 bg-success rounded-full animate-pulse"></div>
            <span className="text-sm text-success">Online</span>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((message) => (
          <div
            key={message.id}
            className={cn(
              "flex items-start space-x-3",
              message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
            )}
          >
            <div className={cn(
              "w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0",
              message.role === 'user' 
                ? 'bg-primary text-primary-foreground' 
                : message.error
                ? 'bg-destructive text-destructive-foreground'
                : 'bg-gradient-to-r from-accent to-secondary text-accent-foreground'
            )}>
              {message.role === 'user' ? <User className="w-4 h-4" /> : 
               message.error ? <AlertCircle className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
            </div>

            <div className={cn(
              "flex-1 space-y-3",
              message.role === 'user' ? 'items-end' : 'items-start'
            )}>
              {/* Markdown message */}
              <div className={cn(
                "inline-block p-4 rounded-2xl max-w-3xl",
                message.role === 'user'
                  ? 'bg-primary text-primary-foreground ml-auto'
                  : message.error
                  ? 'bg-destructive/10 border border-destructive/30'
                  : 'bg-card border border-border/50'
              )}>
                <div className="text-sm leading-relaxed whitespace-pre-wrap">
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                </div>
              </div>

              {/* Render all product cards */}
              {message.searchResults?.search_results?.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {message.searchResults.search_results.map(product => {
                    console.log("Rendering product card:", product);
                    return <ProductCardComponent key={product.id} product={product} />;
                  })}
                </div>
              )}

              <div className="text-xs text-muted-foreground">
                {message.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}

        {isTyping && <TypingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-6 border-t border-border/50 bg-card/30 backdrop-blur-sm">
        <div className="flex space-x-3">
          <div className="flex-1 relative">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me about smartphones, prices, specs, or recommendations..."
              className="pr-12 bg-background/50 border-border/50 focus:border-accent/50 h-12"
              disabled={isTyping}
            />
            <Sparkles className="w-4 h-4 text-accent absolute right-3 top-1/2 transform -translate-y-1/2" />
          </div>
          <Button 
            onClick={handleSend} 
            disabled={!input.trim() || isTyping}
            variant="ai"
            size="icon"
            className="h-12 w-12"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>

        <div className="flex flex-wrap gap-2 mt-3">
          {[
            'Best gaming phones under 30k EGP',
            'iPhone vs Samsung flagship comparison', 
            'Budget phones with good cameras',
            'Latest 5G phones from B.Tech'
          ].map((suggestion) => (
            <Button
              key={suggestion}
              variant="outline"
              size="sm"
              onClick={() => setInput(suggestion)}
              className="text-xs"
            >
              {suggestion}
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
};
