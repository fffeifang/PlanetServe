package io.planetllm;

/**
 * Usage:
 *   ./gradlew run                    - Start the verification server
 */
public class App {
    
    private static final int DEFAULT_PORT = 26658;
    
    public static void main(String[] args) {
        System.out.println("╔═══════════════════════════════════════════════════════════╗");
        System.out.println("║         PlanetServe Verification Application              ║");
        System.out.println("║   Tendermint-based Model Node Verification Protocol       ║");
        System.out.println("╚═══════════════════════════════════════════════════════════╝");
        System.out.println();
        
        // Parse command line arguments
        if (args.length > 0) {
            String command = args[0].toLowerCase();
            switch (command) {
                case "help":
                case "-h":
                case "--help":
                    printHelp();
                    return;

                default:
                    System.err.println("Unknown command: " + command);
                    System.err.println("Note: Standalone test classes removed. Use unified C++ demo instead:");
                    System.err.println("  cd ../.. && ./scripts/run_full_demo.sh");
                    System.exit(1);
            }
        }
        
        // Start the server
        startServer();
    }
    
    private static void startServer() {
        try {
            // Get node ID from environment or use default
            String nodeId = System.getenv("NODE_ID");
            if (nodeId == null || nodeId.isEmpty()) {
                nodeId = "verification_node_1";
            }
            
            // Get port from environment or use default
            int port = DEFAULT_PORT;
            String portEnv = System.getenv("PORT");
            if (portEnv != null && !portEnv.isEmpty()) {
                try {
                    port = Integer.parseInt(portEnv);
                } catch (NumberFormatException e) {
                    System.err.println("Invalid PORT environment variable, using default: " + DEFAULT_PORT);
                }
            }
            
            System.out.println("Configuration:");
            System.out.println("  Node ID: " + nodeId);
            System.out.println("  Port: " + port);
            System.out.println();
            
            // Create the application instance
            CredibilityApp app = new CredibilityApp(nodeId);
            
            // Check LLM server health
            System.out.println("Checking LLM server connection...");
            if (app.checkLLMHealth()) {
                System.out.println("  ✓ LLM server is healthy");
            } else {
                System.out.println("  ⚠ LLM server not available (verification will use fallback)");
            }
            System.out.println();
            
            // Create and start the server
            GrpcServer server = new GrpcServer(app, port);
            server.start();
            
            System.out.println("╔═══════════════════════════════════════════════════════════╗");
            System.out.println("║              Server Started Successfully                  ║");
            System.out.println("╠═══════════════════════════════════════════════════════════╣");
            System.out.println("║  Endpoints:                                               ║");
            System.out.println("║    POST /tx     - Submit transactions                     ║");
            System.out.println("║    GET  /query  - Query application state                 ║");
            System.out.println("║    GET  /commit - Commit state hash                       ║");
            System.out.println("╠═══════════════════════════════════════════════════════════╣");
            System.out.println("║  Press Ctrl+C to stop                                     ║");
            System.out.println("╚═══════════════════════════════════════════════════════════╝");
            System.out.println();
            
            // Keep the application running
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                System.out.println("\nShutting down...");
            }));
            
            while (true) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    break;
                }
            }
            
        } catch (Exception e) {
            System.err.println("Failed to start application: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    private static void printHelp() {
        System.out.println("Usage: ./gradlew run");
        System.out.println();
        System.out.println("Description:");
        System.out.println("  Start the PlanetServe verification server for BFT consensus");
        System.out.println("  Note: Standalone test classes removed. Use unified C++ demo:");
        System.out.println("    cd ../.. && ./scripts/run_full_demo.sh");
        System.out.println();
        System.out.println("Environment Variables:");
        System.out.println("  NODE_ID    Verification node identifier (default: verification_node_1)");
        System.out.println("  PORT       Server port (default: 26658)");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  ./gradlew run");
        System.out.println("  NODE_ID=verifier_2 PORT=26659 ./gradlew run");
    }
}
