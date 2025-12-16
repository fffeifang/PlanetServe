package io.planetllm;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;
import java.util.Map;
import java.util.HashMap;

/**
 * HTTP Server for the Credibility Application
 */
public class GrpcServer {
    private final CredibilityApp app;
    private final int port;
    private HttpServer server;
    private final ObjectMapper objectMapper;

    public GrpcServer(CredibilityApp app, int port) {
        this.app = app;
        this.port = port;
        this.objectMapper = new ObjectMapper();
    }
    
    /**
     * Start the HTTP server
     */
    public void start() throws IOException {
        server = HttpServer.create(new InetSocketAddress(port), 0);
        
        // Set up context handlers for different endpoints
        server.createContext("/tx", new TxHandler());
        server.createContext("/query", new QueryHandler());
        server.createContext("/commit", new CommitHandler());
        
        // Use a thread pool with 10 threads to handle requests
        server.setExecutor(Executors.newFixedThreadPool(10));
        server.start();
        
        System.out.println("HTTP server started on port " + port);
        
        // Add shutdown hook to stop server gracefully
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("Shutting down HTTP server");
            stop();
        }));
    }
    
    /**
     * Stop the HTTP server
     */
    public void stop() {
        if (server != null) {
            server.stop(0);
            System.out.println("HTTP server stopped");
        }
    }
    
    /**
     * Handler for transaction requests
     */
    private class TxHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            // Only handle POST requests
            if (!"POST".equals(exchange.getRequestMethod())) {
                sendResponse(exchange, 405, "Method Not Allowed");
                return;
            }
            
            try {
                // Read request body
                String requestBody = readInputStream(exchange.getRequestBody());
                
                // Process the transaction
                String response = app.deliverTx(requestBody.getBytes(StandardCharsets.UTF_8));
                
                // Send response
                sendResponse(exchange, 200, response);
                
            } catch (Exception e) {
                sendResponse(exchange, 500, "Error processing transaction: " + e.getMessage());
            }
        }
    }
    
    /**
     * Handler for query requests
     */
    private class QueryHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            // Only handle GET requests
            if (!"GET".equals(exchange.getRequestMethod())) {
                sendResponse(exchange, 405, "Method Not Allowed");
                return;
            }
            
            try {
                // Parse query parameters
                String query = exchange.getRequestURI().getQuery();
                Map<String, String> params = parseQueryParams(query);
                
                String path = params.get("path");
                if (path == null) {
                    sendResponse(exchange, 400, "Missing 'path' parameter");
                    return;
                }
                
                // Create data byte array from other parameters if needed
                byte[] data = null;
                if (params.containsKey("data")) {
                    data = params.get("data").getBytes(StandardCharsets.UTF_8);
                }
                
                // Process the query
                String response = app.query(path, data);
                
                // Send response
                sendResponse(exchange, 200, response);
                
            } catch (Exception e) {
                sendResponse(exchange, 500, "Error processing query: " + e.getMessage());
            }
        }
    }
    
    /**
     * Handler for commit requests
     */
    private class CommitHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            // Only handle GET requests for simplicity
            if (!"GET".equals(exchange.getRequestMethod())) {
                sendResponse(exchange, 405, "Method Not Allowed");
                return;
            }
            
            try {
                // Process the commit
                byte[] hash = app.commit();
                
                // Encode hash as hex string
                String hashHex = bytesToHex(hash);
                
                // Send response
                sendResponse(exchange, 200, "Committed state with hash: " + hashHex);
                
            } catch (Exception e) {
                sendResponse(exchange, 500, "Error committing state: " + e.getMessage());
            }
        }
    }
    
    /**
     * Helper method to send HTTP response
     */
    private void sendResponse(HttpExchange exchange, int statusCode, String response) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(statusCode, response.length());
        
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(response.getBytes(StandardCharsets.UTF_8));
        }
    }
    
    /**
     * Helper method to read input stream to string
     */
    private String readInputStream(InputStream is) throws IOException {
        StringBuilder sb = new StringBuilder();
        byte[] buffer = new byte[1024];
        int length;
        while ((length = is.read(buffer)) != -1) {
            sb.append(new String(buffer, 0, length, StandardCharsets.UTF_8));
        }
        return sb.toString();
    }
    
    /**
     * Helper method to parse query parameters
     */
    private Map<String, String> parseQueryParams(String query) {
        Map<String, String> params = new HashMap<>();
        if (query != null) {
            String[] pairs = query.split("&");
            for (String pair : pairs) {
                int idx = pair.indexOf("=");
                if (idx > 0) {
                    params.put(pair.substring(0, idx), pair.substring(idx + 1));
                }
            }
        }
        return params;
    }
    
    /**
     * Helper method to convert bytes to hex string
     */
    private String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }
} 