import { Switch, Route, Link } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import HomePage from "@/pages/HomePage";
import TranslationPage from "@/pages/TranslationPage";

import WebSearchPage from "@/pages/WebSearchPage";

import { AnalyticsPage } from "@/pages/AnalyticsPage";
import NotFound from "@/pages/not-found";
import { BrainCircuit, Languages, FileEdit, Globe, Bot, Brain, Mail, User, LogOut, Trash2, FileText } from "lucide-react";
import { AuthProvider, useAuth } from "@/hooks/use-auth";
import { useState, createContext, useContext } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CreditBalance } from "@/components/CreditBalance";

// Reset Context
interface ResetContextType {
  resetAll: () => void;
}

const ResetContext = createContext<ResetContextType | null>(null);

export function useReset() {
  const context = useContext(ResetContext);
  if (!context) {
    throw new Error("useReset must be used within a ResetProvider");
  }
  return context;
}

function LoginDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { loginMutation, registerMutation } = useAuth();
  const [activeTab, setActiveTab] = useState("login");
  const [loginForm, setLoginForm] = useState({ username: "", password: "" });
  const [registerForm, setRegisterForm] = useState({ username: "", password: "", email: "" });

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    loginMutation.mutate(loginForm, {
      onSuccess: () => {
        onOpenChange(false);
        setLoginForm({ username: "", password: "" });
      }
    });
  };

  const handleRegister = (e: React.FormEvent) => {
    e.preventDefault();
    registerMutation.mutate(registerForm, {
      onSuccess: () => {
        onOpenChange(false);
        setRegisterForm({ username: "", password: "", email: "" });
      }
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Account Access</DialogTitle>
        </DialogHeader>
        
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="login">Login</TabsTrigger>
            <TabsTrigger value="register">Register</TabsTrigger>
          </TabsList>
          
          <TabsContent value="login" className="space-y-4">
            <form onSubmit={handleLogin} className="space-y-4">
              <div>
                <Label htmlFor="login-username">Username</Label>
                <Input
                  id="login-username"
                  type="text"
                  value={loginForm.username}
                  onChange={(e) => setLoginForm({ ...loginForm, username: e.target.value })}
                  required
                  autoComplete="username"
                  data-testid="input-login-username"
                />
              </div>
              <div>
                <Label htmlFor="login-password">
                  Password{loginForm.username.toLowerCase().trim() === "jmkuczynski" ? " (Optional for JMKUCZYNSKI)" : ""}
                </Label>
                <Input
                  id="login-password"
                  type="password"
                  value={loginForm.password}
                  onChange={(e) => setLoginForm({ ...loginForm, password: e.target.value })}
                  required={loginForm.username.toLowerCase().trim() !== "jmkuczynski"}
                  autoComplete="current-password"
                  data-testid="input-login-password"
                />
              </div>
              <Button type="submit" className="w-full" disabled={loginMutation.isPending} data-testid="button-login">
                {loginMutation.isPending ? "Logging in..." : "Login"}
              </Button>
            </form>
          </TabsContent>
          
          <TabsContent value="register" className="space-y-4">
            <form onSubmit={handleRegister} className="space-y-4">
              <div>
                <Label htmlFor="register-username">Username</Label>
                <Input
                  id="register-username"
                  type="text"
                  value={registerForm.username}
                  onChange={(e) => setRegisterForm({ ...registerForm, username: e.target.value })}
                  required
                  data-testid="input-register-username"
                />
              </div>
              <div>
                <Label htmlFor="register-email">Email (optional)</Label>
                <Input
                  id="register-email"
                  type="email"
                  value={registerForm.email}
                  onChange={(e) => setRegisterForm({ ...registerForm, email: e.target.value })}
                  data-testid="input-register-email"
                />
              </div>
              <div>
                <Label htmlFor="register-password">Password</Label>
                <Input
                  id="register-password"
                  type="password"
                  value={registerForm.password}
                  onChange={(e) => setRegisterForm({ ...registerForm, password: e.target.value })}
                  required
                  data-testid="input-register-password"
                />
              </div>
              <Button type="submit" className="w-full" disabled={registerMutation.isPending} data-testid="button-register">
                {registerMutation.isPending ? "Creating account..." : "Create Account"}
              </Button>
            </form>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}

function ResetConfirmDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { resetAll } = useReset();

  const handleReset = () => {
    resetAll();
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Reset All Data</DialogTitle>
          <DialogDescription>
            This will clear all your current input and analysis results. You'll start completely fresh. This action cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <div className="flex gap-3 justify-end">
          <Button variant="outline" onClick={() => onOpenChange(false)} data-testid="button-cancel-reset">
            Cancel
          </Button>
          <Button variant="destructive" onClick={handleReset} data-testid="button-confirm-reset">
            <Trash2 className="h-4 w-4 mr-2" />
            Reset All
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function Navigation() {
  const { user, logoutMutation } = useAuth();
  const [loginDialogOpen, setLoginDialogOpen] = useState(false);
  const [resetDialogOpen, setResetDialogOpen] = useState(false);

  return (
    <nav className="bg-primary text-primary-foreground py-4">
      <div className="container mx-auto flex justify-between items-center">
        <div className="flex items-center gap-6">
          <div className="flex flex-col items-center">
            <div className="font-bold text-xl">MEMO PRO</div>
            <FileText className="h-4 w-4" />
          </div>
          <a 
            href="mailto:contact@zhisystems.ai" 
            className="flex items-center gap-2 hover:underline text-sm"
          >
            <Mail className="h-4 w-4" />
            <span>Contact Us</span>
          </a>
        </div>
        <div className="flex items-center gap-6">
          <ul className="flex gap-6">
            <li>
              <Link href="/" className="flex items-center gap-2 hover:underline">
                <BrainCircuit className="h-5 w-5" />
                <span>Intelligence Analysis</span>
              </Link>
            </li>
            <li>
              <Link href="/analytics" className="flex items-center gap-2 hover:underline">
                <Brain className="h-5 w-5" />
                <span>Cognitive Analytics</span>
              </Link>
            </li>
          </ul>
          
          <div className="flex items-center gap-3">
            {user && (
              <div className="bg-primary-foreground/10 px-3 py-1.5 rounded-md">
                <CreditBalance />
              </div>
            )}
            
            <Button 
              variant="ghost" 
              size="sm"
              onClick={() => setResetDialogOpen(true)}
              className="text-primary-foreground hover:bg-primary-foreground/10"
              data-testid="button-reset-all"
            >
              <Trash2 className="h-4 w-4 mr-1" />
              Reset All
            </Button>
            
            <div className="flex items-center gap-4 border-l border-primary-foreground/20 pl-4">
              {user ? (
                <div className="flex items-center gap-2">
                  <span className="text-sm">Welcome, {user.username}!</span>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    onClick={() => logoutMutation.mutate()}
                    className="text-primary-foreground hover:bg-primary-foreground/10"
                    data-testid="button-logout"
                  >
                    <LogOut className="h-4 w-4 mr-1" />
                    Logout
                  </Button>
                </div>
              ) : (
                <Button 
                  variant="ghost" 
                  size="sm"
                  onClick={() => setLoginDialogOpen(true)}
                  className="text-primary-foreground hover:bg-primary-foreground/10"
                  data-testid="button-open-login"
                >
                  <User className="h-4 w-4 mr-1" />
                  Login / Register
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
      
      <LoginDialog open={loginDialogOpen} onOpenChange={setLoginDialogOpen} />
      <ResetConfirmDialog open={resetDialogOpen} onOpenChange={setResetDialogOpen} />
    </nav>
  );
}

function Router({ resetKey }: { resetKey: number }) {
  return (
    <>
      <Navigation />
      <Switch key={resetKey}>
        <Route path="/" component={HomePage} />
        <Route path="/analytics" component={AnalyticsPage} />

        <Route component={NotFound} />
      </Switch>
    </>
  );
}

function App() {
  const [resetKey, setResetKey] = useState(0);

  const resetAll = () => {
    // Clear app-specific localStorage (preserve auth and theme)
    const keysToRemove: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith('cap:')) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach(key => localStorage.removeItem(key));
    
    // Remount Router to reset all component state
    setResetKey(prev => prev + 1);
  };

  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <ResetContext.Provider value={{ resetAll }}>
          <TooltipProvider>
            <Toaster />
            <Router resetKey={resetKey} />
          </TooltipProvider>
        </ResetContext.Provider>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;
