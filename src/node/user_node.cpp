#include "user_node.hpp"
#include <iostream>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <fstream>
#include <sstream>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <random> 
#include "../encrypt_p2p/network_handler.hpp"
namespace node {

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t totalSize = size * nmemb;
    std::string* response = static_cast<std::string*>(userp);
    response->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

UserNode::UserNode(std::string ip_address, int port) : listener(IP_ADDRESS, PORT) {
    // Constructor implementation
    IP_ADDRESS = ip_address;
    PORT = port;
    std::pair<std::string, std::string> keyPair = encrypt_p2p::generateRSAKeyPair();
    RSA_PUBLIC_KEY = keyPair.first;
    RSA_PRIVATE_KEY = keyPair.second;
    if (!listener.bind("0.0.0.0", PORT)) { //TODO: check later
        std::cerr << "Failed to bind listener socket to port " << PORT << std::endl;
        return;
    }
    std::cout << "Listener socket bound to port " << PORT << std::endl;
   
    std::ifstream ifs("../../datasets/chat/dev-v2.0.json");
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file: ../../datasets/chat/dev-v2.0.json");
    }

    nlohmann::json squadData;
    ifs >> squadData;
    ifs.close();

    for (const auto& item : squadData["data"]) {
        for (const auto& paragraph : item["paragraphs"]) {
            for (const auto& question : paragraph["qas"]) {
                allQuestions.push_back(question["question"]);
            } if (!squadData.contains("data") || !squadData["data"].is_array()) {
        throw std::runtime_error("Invalid SQuAD JSON format: missing 'data' array");
    }

    for (const auto& article : squadData["data"]) {
        // each "article" typically has "title" and "paragraphs"
        if (!article.contains("paragraphs") || !article["paragraphs"].is_array()) {
            continue; // skip if format doesn't match
        }

        for (const auto& paragraph : article["paragraphs"]) {
            // paragraph has "qas" (an array of question/answer sets)
            if (!paragraph.contains("qas") || !paragraph["qas"].is_array()) {
                continue;
            }

            for (const auto& qa : paragraph["qas"]) {
                // "qa" contains "question", "id", "answers", etc.
                if (qa.contains("question") && qa["question"].is_string()) {
                    std::string question = qa["question"].get<std::string>();
                    allQuestions.push_back(question);
                }
            }
        }
        }
    }
    }
    StartHandleMessage();
}

UserNode::~UserNode() {
    // Destructor implementation
    StopHandleMessage();
}

std::string UserNode::httpPost(const std::string &url, const std::string &jsonPayload) {
    CURL *curl = curl_easy_init();
    std::string responseString;
    
    if (curl) {
        struct curl_slist *headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonPayload.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseString);
        
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "[UserNode] curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
        
        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
    }
    
    return responseString;
}


void UserNode::initialize() {
    std::cout << "UserNode initializing..." << std::endl;
    // register to the verification node (add IP address and RSA public key to the verification node)
    // get the list of nodes in the same region from verification node
    // read verification node list from config file (../../datasets/verification_node_list.txt)
    std::ifstream file("../../datasets/verification_node_list.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening file: ../../datasets/verification_node_list.txt" << std::endl;
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        VerificationNodeList.push_back(line);
    }
    file.close();
    // choose one verification node near this node based on IP address
    for (const auto &node : VerificationNodeList) {
        if (node.find(IP_ADDRESS) != std::string::npos) {
            IP_verification_node = node;
            break;
        }
    }
    // register to the verification node
    // TODO: send RSA public key to the verification node using http post
    // TODO: receive the list of nodes in the same region from the verification node
    std::string url = "http://" + IP_verification_node + ":8080/register";
    std::string payload = "{\"ip_address\": \"" + IP_ADDRESS + "\", \"port\": \"" + std::to_string(PORT) + "\", \"rsa_public_key\": \"" + RSA_PUBLIC_KEY + "\"}";
    std::string response = this->httpPost(url, payload);
    std::cout << "Response from verification node: " << response << std::endl;
}

std::string UserNode::GenerateSessionID(){ // 160 bits sessionID and UUID space (128-bit)
    // generate a new sessionID, unique accross all nodes
    boost::uuids::random_generator generator;
    boost::uuids::uuid id = generator();
    std::string to_hash = IP_ADDRESS + ":" + std::to_string(PORT) + boost::uuids::to_string(id);
    // Hash using SHA-1 (160 bits)
    boost::uuids::detail::sha1 sha;
    sha.process_bytes(to_hash.data(), to_hash.size());
    unsigned int hash[5] = {0};
    sha.get_digest(hash);

    // Convert hash to string
    std::string hash_string;
    for (int i = 0; i < 5; ++i) {
        hash_string += std::to_string(hash[i]);
    }

    return hash_string;
}

std::vector<Message> UserNode::CreateSession(int N){
    // create a new session, TODO: confirm datasets
    if (N >= static_cast<int>(allQuestions.size())) {
        std::cerr << "N is greater than the number of questions" << std::endl;
        return std::vector<Message>();
    }

    // Otherwise, shuffle and pick first n
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(allQuestions.begin(), allQuestions.end(), g);

    // Create a new vector with n questions
    std::vector<std::string> result(allQuestions.begin(), allQuestions.begin() + N);
    std::string sessionID = GenerateSessionID();
    std::vector<Message> session;
    int seq_num = 0;
    for (const auto& question : result) {
        Message message;
        message.sessionID = sessionID;
        message.seq_num = seq_num;
        message.question = question;
        if (question == result.back()) {
            message.is_last_message = true;
        }
        else{
            message.is_last_message = false;
        }
        session.push_back(message);
        seq_num++;
    }
    return session;
}

bool UserNode::EstablishProxyConnection(std::string &sessionID, std::vector<std::vector<std::pair<std::string, int>>> &proxy_IP_path, std::string &model_IP_address, int model_port){
    // try to establish N proxy connections
    // using ZMQ and onion routing
    // TODO: may need to run in a thread to listen to confirmation message?
    
    std::string probe_message = model_IP_address;
    std::string public_key;
    std::string encrypted_probe_message = probe_message;
    std::vector<std::string> encrypted_probe_message_tosend;
    // prepare onion routing
    for (int i = 0; i < proxy_IP_path.size(); i++){
        for (int j = proxy_IP_path[i].size() - 1; j >= 0; j--){
            // find the public key of the proxy node using user node list
            for (const auto &node : UserNodesList){
                if (std::get<0>(node) == proxy_IP_path[i][j].first && std::get<1>(node) == proxy_IP_path[i][j].second){
                    public_key = std::get<2>(node);
                    break;
                }
            }
            
            if(j != proxy_IP_path[i].size() - 1){ // not the last hop, encrypt next hop
                //do not encrypt sessionID
                encrypted_probe_message = "P| " + sessionID + " " + encrypt_p2p::encryptRSA(proxy_IP_path[i][j + 1].first + " " + std::to_string(proxy_IP_path[i][j + 1].second) + " " + encrypted_probe_message, public_key); 
            }
            else{
                encrypted_probe_message = "H| " + sessionID + " " + encrypt_p2p::encryptRSA(encrypted_probe_message, public_key);
            }

        } 
        encrypted_probe_message_tosend.push_back(encrypted_probe_message);
    }
    // send the encrypted probe message to the next hop using ZMQ

     ProxyNodeList.push_back(std::make_tuple(
        sessionID, 
        std::make_pair("null", 0), 
        std::make_pair("null", 0), 
        std::make_pair("null", 0), 
        std::make_pair("null", 0)
    ));
    
    for (int i = 0; i < encrypted_probe_message_tosend.size(); i++){
       encrypt_p2p::NetworkHandler handler(IP_ADDRESS, PORT); 
       handler.connect(proxy_IP_path[i][0].first, proxy_IP_path[i][0].second);
       handler.sendData(encrypted_probe_message_tosend[i]);
       handler.disconnect();
    }

    // Wait for confirmation messages from all proxy paths
    const int MAX_WAIT_TIME_MS = 10000; // 10 seconds, remember to change this
    const int SLEEP_INTERVAL_MS = 100;
    int elapsed_time = 0;
    bool all_confirmed = false;
    
    while (elapsed_time < MAX_WAIT_TIME_MS) {
        // Check if all proxy connections are confirmed
        all_confirmed = true;
        
        // Find the ProxyNodeList entry for this session
        for (const auto &proxy_entry : ProxyNodeList) {
            if (std::get<0>(proxy_entry) == sessionID) {
                // Check if any of the required proxy nodes are still null
                // If we need all 4 proxy nodes
                if (std::get<1>(proxy_entry) == std::make_pair(std::string("null"), 0) ||
                    std::get<2>(proxy_entry) == std::make_pair(std::string("null"), 0) ||
                    std::get<3>(proxy_entry) == std::make_pair(std::string("null"), 0) ||
                    std::get<4>(proxy_entry) == std::make_pair(std::string("null"), 0)) {
                    all_confirmed = false;
                    break;
                }
            }
        }
        
        if (all_confirmed) {
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_INTERVAL_MS));
        elapsed_time += SLEEP_INTERVAL_MS;
    }
    
    if (!all_confirmed){
        std::cerr << "Failed to establish all proxy connections" << std::endl;
        return false;
    }
    return true;
}
    
void UserNode::StartHandleMessage() {
    running = true;
    receiverThread = std::thread(&UserNode::receiveLoop, this);
    processorThread = std::thread(&UserNode::processMessages, this); 
}

void UserNode::StopHandleMessage() {
    running = false;
    cv.notify_all();  // exit all threads
}

void UserNode::receiveLoop() {
    while (running) {
        std::string sender_IP, msg;
        int sender_port;
        msg = listener.receiveData(sender_IP, sender_port, 100); // poll with 100ms timeout
        if (!msg.empty()) {
            std::lock_guard<std::mutex> lock(mtx);
            messageQueue.push(std::make_tuple(sender_IP, sender_port, msg));
            cv.notify_one();
        }
        // TODO: may need to sleep ?
    }
}


void UserNode::processMessages() {
        while (running) {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return !messageQueue.empty() || !running; });
            while (!messageQueue.empty()) {
                std::string sender_IP, msg, whole_message;
                int sender_port;
                std::tie(sender_IP, sender_port, whole_message) = messageQueue.front();
                messageQueue.pop();
                lock.unlock();
                std::istringstream iss_tmp(whole_message);
                std::string message_type, sessionID;
                iss_tmp >> message_type >> sessionID >> msg;
                //identify ForwardRegularMessage

                bool found = false;
                std::string successor_hop_IP;
                int successor_hop_port;
                std::string predecessor_hop_IP;
                int predecessor_hop_port;
                for (const auto &state : RelayStateTable){
                    if (std::get<0>(state) == sessionID){
                        // from start node to proxy node
                        successor_hop_IP = std::get<3>(state);
                        successor_hop_port = std::get<4>(state);
                        predecessor_hop_IP = std::get<1>(state);
                        predecessor_hop_port = std::get<2>(state);
                        found = true;
                    }
                }
                bool is_forward = false;
                if (message_type == "R|"){
                    if (found){
                        if (sender_IP == predecessor_hop_IP && sender_port == predecessor_hop_port){
                            is_forward = ForwardRegularMessage(sessionID, successor_hop_IP, successor_hop_port, msg);
                        }
                        else if (sender_IP == successor_hop_IP && sender_port != successor_hop_port){
                            is_forward = ForwardRegularMessage(sessionID, predecessor_hop_IP, predecessor_hop_port, msg);
                        }
                    }
                    else{
                        std::cerr << "No next hop found for session: " << sessionID << std::endl;
                    }
                }
                else if (message_type == "P|"){
                    if (found){
                        std::cerr << "exist sessionID in relay state table for probe message" << std::endl;
                    }
                    else{
                        
                        is_forward = ForwardProbeMessage(sender_IP, sender_port, sessionID, msg);
                    }
                }
                else if (message_type == "H|"){
                    if (found){
                        std::cerr << "exist sessionID in relay state table for acting as a proxy node for probe message" << std::endl; 
                    }
                    else{
                        is_forward = HandleProbeMessage(sender_IP, sender_port, sessionID, msg);
                    }
                }
                else if (message_type == "C|"){
                    if (found){
                        is_forward = ForwardConfirmationMessage(predecessor_hop_IP, predecessor_hop_port, sessionID);
                    }
                    else{
                        if (std::find(ActiveSessionTable.begin(), ActiveSessionTable.end(), sessionID) != ActiveSessionTable.end()){
                            is_forward = HandleConfirmationMessage(sender_IP, sender_port, sessionID);
                        }
                        else{
                            std::cerr << "sessionID not found in active session table" << std::endl;
                        }
                    }
                }
                else{
                    std::cerr << "Unknown message type: " << message_type << std::endl;
                }
                lock.lock();
            }
        }
    }

bool UserNode::ForwardRegularMessage(std::string &sessionID, std::string &next_hop_ip, int next_hop_port, std::string &encrypted_message){
    // forward message to the next hop after establishing proxy connection
    // using relay state table to find the next hop
    encrypt_p2p::NetworkHandler handler(IP_ADDRESS, PORT);
    handler.connect(next_hop_ip, next_hop_port);
    handler.sendData(encrypted_message);
    handler.disconnect(); // TODO: need to check if this is correct (success or not)
    return true;
}

bool UserNode::ForwardProbeMessage(std::string &predecessor_node_IP, int predecessor_node_port, std::string &sessionID, std::string &msg){
    // forward probe message to the next hop for establishing proxy connection
    msg = encrypt_p2p::encryptRSA(msg, RSA_PRIVATE_KEY);
    std::istringstream iss(msg);
    std::string next_hop_IP, next_hop_port, encrypted_message;
    iss >> next_hop_IP >> next_hop_port >> encrypted_message;
    encrypt_p2p::NetworkHandler handler(IP_ADDRESS, PORT);
    handler.connect(next_hop_IP, std::stoi(next_hop_port));
    handler.sendData(encrypted_message);
    handler.disconnect(); // TODO: need to check if this is correct (success or not)
    RelayStateTable.push_back(std::make_tuple(sessionID, predecessor_node_IP, predecessor_node_port, next_hop_IP, std::stoi(next_hop_port)));
    return true;
}

bool UserNode::HandleProbeMessage(std::string &predecessor_node_IP, int predecessor_node_port, std::string &sessionID, std::string &model_IP_address){
    // act as a proxy node for probe message
    // using relay state table to find the next hop
    encrypt_p2p::NetworkHandler handler(IP_ADDRESS, PORT);
    // TODO:change
    handler.connect(model_IP_address, MODEL_NODE_PORT);
    handler.sendData(sessionID);
    handler.disconnect(); // TODO: need to check if this need long connection or not
    ConversationList_proxy.push_back(std::make_tuple(sessionID, model_IP_address, false));
    // send confirmation message to the predecessor node
    std::string confirmation_message = "C| " + sessionID ;
    handler.connect(predecessor_node_IP, predecessor_node_port);
    handler.sendData(confirmation_message);
    handler.disconnect();
    return true;
}

bool UserNode::HandleConfirmationMessage(std::string &sender_node_IP, int sender_node_port, std::string &sessionID){
    //find sender_node_IP and sender_node_port in proxy_IP_path
    for (const auto &paths : proxy_IP_path){
        if (paths.first == sessionID){
            for (const auto &path : paths.second){
                if (sender_node_IP == path[0].first && sender_node_port == path[0].second){
                    bool found = false;
                    for (int i = 0; i < ProxyNodeList.size(); i++) {
                    if (std::get<0>(ProxyNodeList[i]) == sessionID) {
                        for (int j = 0; j < 4; j++) {
                            switch (j) {
                                case 0:
                                    if (std::get<1>(ProxyNodeList[i]) != std::make_pair(std::string("null"), 0)) {
                                        std::get<1>(ProxyNodeList[i]) = std::make_pair(path[1].first, path[1].second);
                                        found = true;
                                    }
                            
                                    break;
                                case 1:
                                    if (std::get<2>(ProxyNodeList[i]) != std::make_pair(std::string("null"), 0)) {
                                        std::get<2>(ProxyNodeList[i]) = std::make_pair(path[2].first, path[2].second);
                                        found = true;
                                    }
                                    break;
                                case 2:
                                    if (std::get<3>(ProxyNodeList[i]) != std::make_pair(std::string("null"), 0)) {
                                        std::get<3>(ProxyNodeList[i]) = std::make_pair(path[3].first, path[3].second);
                                        found = true;
                                    }
                                    break;
                                case 3:
                                    if (std::get<4>(ProxyNodeList[i]) != std::make_pair(std::string("null"), 0)) {
                                        std::get<4>(ProxyNodeList[i]) = std::make_pair(path[4].first, path[4].second);
                                        found = true;
                                    }
                                    break;
                            }
                        }
                    }
                    }
                    if (!found){
                        return false;
                    }
                    return true;
                }
            }
        }
    }
    return false;
 }
 
std::string UserNode::SendSession(std::vector<Message> &session, std::string model_IP_address, int n, int k){
    // send message to the model node
    ActiveSessionTable.push_back(session[0].sessionID);
    // For each message in the session
    for (const auto& message : session) {
        // Create JSON representation of the message
        std::stringstream ss;
        // TODO: may need to change
        ss << "{"
           << "\"sessionID\":\"" << message.sessionID << "\","
           << "\"seq_num\":" << message.seq_num << ","
           << "\"question\":\"" << message.question << "\","
           << "\"proxy_list\":\"" << message.proxy_list[0] + " " + message.proxy_list[1] + " " + message.proxy_list[2] + " " + message.proxy_list[3] << "\","
           << "\"is_last_message\":" << (message.is_last_message ? "true" : "false")
           << "}";
        std::string messageJson = ss.str();
        
        // Get proxy paths for this session
        std::vector<std::pair<std::string, int>> paths;
        for (const auto& entry : ProxyNodeList) {
            if (std::get<0>(entry) == session[0].sessionID) {
                // Add all non-null proxy paths
                if (std::get<1>(entry) != std::make_pair(std::string("null"), 0)) {
                    paths.push_back(std::get<1>(entry));
                }
                if (std::get<2>(entry) != std::make_pair(std::string("null"), 0)) {
                    paths.push_back(std::get<2>(entry));
                }
                if (std::get<3>(entry) != std::make_pair(std::string("null"), 0)) {
                    paths.push_back(std::get<3>(entry));
                }
                if (std::get<4>(entry) != std::make_pair(std::string("null"), 0)) {
                    paths.push_back(std::get<4>(entry));
                }
                break;
            }
        }
        
        // If we don't have enough paths, return error
        if (paths.size() < static_cast<size_t>(n)) {
            std::cerr << "Not enough proxy paths for S-IDA. Need " << n << " but only have " << paths.size() << std::endl;
            return "ERROR: Not enough proxy paths for S-IDA";
        }
        
        // Send the message using S-IDA
        if (!SendSIDAMessage(session[0].sessionID, message.seq_num, messageJson, paths, n, k)) {
            std::cerr << "Failed to send message using S-IDA" << std::endl;
            return "ERROR: Failed to send message using S-IDA";
        }
    }
    
    return "SUCCESS"; 
}

bool UserNode::SendWithConnection(const std::string& target_ip, int target_port, const std::string& data) {
    // Connection key 
    std::string connKey = target_ip + ":" + std::to_string(target_port);
    bool result = false;
    
    {
        std::lock_guard<std::mutex> lock(connectionMtx);
        // Get or create connection
        if (connectionPool.find(connKey) == connectionPool.end()) {
            // Create new connection
            connectionPool.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(connKey),
                                   std::forward_as_tuple(IP_ADDRESS, PORT));
            if (!connectionPool[connKey].connect(target_ip, target_port)) {
                connectionPool.erase(connKey);
                return false;
            }
        }
        
        // Send data using existing connection
        result = connectionPool[connKey].sendData(data);
    }
    
    return result;
}

void UserNode::CleanupIdleConnections() {
    std::lock_guard<std::mutex> lock(connectionMtx);
    // TODO: implement this
}

bool UserNode::SendSIDAMessage(const std::string& sessionID, int seq_num, const std::string& message,
                               const std::vector<std::pair<std::string, int>>& paths, int n, int k) {
    // Ensure we have enough paths
    if (paths.size() < static_cast<size_t>(n)) {
        std::cerr << "Not enough proxy paths for S-IDA. Need " << n << " but only have " << paths.size() << std::endl;
        return false;
    }
    
    // Apply S-IDA to split the message
    std::vector<encrypt_p2p::SIDA::Clove> cloves;
    try {
        cloves = encrypt_p2p::SIDA::split(message, n, k);
    } catch (const std::exception& e) {
        std::cerr << "S-IDA split failed: " << e.what() << std::endl;
        return false;
    }
    
    // Send each clove through a different path
    bool allSent = true;
    for (size_t i = 0; i < cloves.size(); i++) {
        // Serialize the clove
        std::string serializedClove = encrypt_p2p::SIDA::serializeClove(cloves[i]);
        
        // Create the message format: "R|sessionID|seq_num|clove"
        std::stringstream ss;
        ss << "R|" << sessionID << "|" << seq_num << "|" << serializedClove;
        std::string message = ss.str();
        
        // Forward the message through the proxy
        const auto& path = paths[i];
        if (!SendWithConnection(path.first, path.second, message)) {
            std::cerr << "Failed to send S-IDA clove through path " << i << std::endl;
            allSent = false;
        }
    }
    
    return allSent;
}

bool UserNode::ProcessSIDAClove(const std::string& sessionID, int seq_num, const std::string& serializedClove) {
    // Deserialize the clove
    encrypt_p2p::SIDA::Clove clove;
    try {
        clove = encrypt_p2p::SIDA::deserializeClove(serializedClove);
    } catch (const std::exception& e) {
        std::cerr << "Failed to deserialize S-IDA clove: " << e.what() << std::endl;
        return false;
    }
    
    // Add the clove to the cache
    {
        std::lock_guard<std::mutex> lock(sidaCacheMtx);
        sidaMessageCache[sessionID][seq_num].push_back(clove);
    }
    
    return true;
}

std::string UserNode::ReconstructSIDAMessage(const std::string& sessionID, int seq_num, int k) {
    std::vector<encrypt_p2p::SIDA::Clove> cloves;
    
    // Get the cloves from the cache
    {
        std::lock_guard<std::mutex> lock(sidaCacheMtx);
        if (sidaMessageCache.find(sessionID) == sidaMessageCache.end() ||
            sidaMessageCache[sessionID].find(seq_num) == sidaMessageCache[sessionID].end() ||
            sidaMessageCache[sessionID][seq_num].size() < static_cast<size_t>(k)) {
            
            // Not enough cloves to reconstruct
            return "";
        }
        
        cloves = sidaMessageCache[sessionID][seq_num];
    }
    
    // Reconstruct the message
    try {
        return encrypt_p2p::SIDA::combine(cloves, k);
    } catch (const std::exception& e) {
        std::cerr << "Failed to reconstruct S-IDA message: " << e.what() << std::endl;
        return "";
    }
}

std::string UserNode::ReceiveRegularMessage(std::string &sessionID, int seq_num, int n, int k) {
    
    {
        std::lock_guard<std::mutex> lock(sidaCacheMtx);
        if (sidaMessageCache.find(sessionID) != sidaMessageCache.end() &&
            sidaMessageCache[sessionID].find(seq_num) != sidaMessageCache[sessionID].end() &&
            sidaMessageCache[sessionID][seq_num].size() >= static_cast<size_t>(k)) {
            
            // We have enough fragments, reconstruct the message
            return ReconstructSIDAMessage(sessionID, seq_num, k);
        }
    }
    
    // Not enough fragments yet, return empty string
    return "";
}

} // namespace node