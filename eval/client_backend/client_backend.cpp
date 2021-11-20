// based on https://ncona.com/2019/04/building-a-simple-server-with-cpp/
// based on https://gitlab.com/palisade/palisade-development/-/blob/master/src/pke/examples/simple-real-numbers.cpp
#define PROFILE

#include <sys/socket.h> // For socket functions
#include <netinet/in.h> // For sockaddr_in
#include <cstdlib> // For exit() and EXIT_FAILURE
#include <iostream> // For cout
#include <unistd.h> // For read
#include <netdb.h>
#include <arpa/inet.h>
#include "palisade.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "scheme/ckks/ckks-ser.h"
#include "pubkeylp-ser.h"

using namespace lbcrypto;

// Save-Load locations for keys
const std::string DATAFOLDER = "demoData";
std::string ccLocation = "/cryptocontext.txt";
std::string pubKeyLocation = "/key_pub.txt";    // Pub key
std::string multKeyLocation = "/key_mult.txt";  // relinearization key
std::string rotKeyLocation = "/key_rot.txt";    // automorphism / rotation key
std::string sumKeyLocation = "/key_sum.txt";
std::string cipherLocation = "/ciphertext.txt";

const std::string FINALDATAFOLDER = "finalData";
std::string cipherFinalLocation = "/ciphertext";
std::string ccFinalLocation = "/cryptocontext";

std::vector<double> extractImageData(std::string data){
    std::vector<double> image;
    image.reserve(25*25);
    std::string number = "";
    for (char const &c: data) {
      if (c == ','){
        image.push_back(std::stod(number));
        number = "";
      }
      else{
        if (isdigit(c) || (c == '.')){
          number += c;
        }
      }
    }
    for (auto i : image) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;
    return image;
}

std::tuple<LPKeyPair<DCRTPoly>, CryptoContext<DCRTPoly>, uint>
ckksEncrypt(std::vector<double> input) {
  uint32_t multDepth = 5;
  uint32_t scaleFactorBits = 50;
  uint32_t batchSize = 1024;
  SecurityLevel securityLevel = HEStd_128_classic;

  CryptoContext<DCRTPoly> cc =
      CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
          multDepth, scaleFactorBits, batchSize, securityLevel);

  std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

  // Enable the features that you wish to use
  cc->Enable(ENCRYPTION);
  cc->Enable(SHE);
  cc->Enable(LEVELEDSHE);

  // B. Step 2: Key Generation
  LPKeyPair<DCRTPoly> keys = cc->KeyGen();
  cc->EvalMultKeyGen(keys.secretKey);
  cc->EvalSumKeyGen(keys.secretKey);

  int rotSize = 16;
  std::vector<int32_t> indexList(rotSize);
  std::iota (std::begin(indexList), std::end(indexList), -rotSize);
  cc->EvalAtIndexKeyGen(keys.secretKey, indexList);
  // cc->EvalAtIndexKeyGen(keys.secretKey, {0, -1, -2, -3, -4});

  // Step 3: Encoding and encryption of inputs

  // Inputs
  vector<double> x1 = input;

  // Encoding as plaintexts
  Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

  // Encrypt the encoded vectors
  auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

  // serialize data
  if (!Serial::SerializeToFile(DATAFOLDER + ccLocation, cc, SerType::BINARY)) {
    std::cerr << "Error writing serialization of the crypto context to cryptocontext.txt" << std::endl;
    return std::make_tuple(keys, cc, -1);
  }
  std::cout << "\tCryptocontext serialized" << std::endl;

  if (!Serial::SerializeToFile(DATAFOLDER + pubKeyLocation, keys.publicKey, SerType::BINARY)) {
    std::cerr << "Exception writing public key to pubkey.txt" << std::endl;
    return std::make_tuple(keys, cc, -1);
  }
  std::cout << "\tPublic key serialized" << std::endl;

  std::ofstream multKeyFile(DATAFOLDER + multKeyLocation, std::ios::out | std::ios::binary);
  if (multKeyFile.is_open()) {
    if (!cc->SerializeEvalMultKey(multKeyFile, SerType::BINARY)) {
      std::cerr << "Error writing eval mult keys" << std::endl;
      return std::make_tuple(keys, cc, -1);
    }
    std::cout << "\tEvalMult relinearization keys have been serialized" << std::endl;
    multKeyFile.close();
  } else {
    std::cerr << "Error serializing EvalMult keys" << std::endl;
    return std::make_tuple(keys, cc, -1);
  }

  std::ofstream rotationKeyFile(DATAFOLDER + rotKeyLocation, std::ios::out | std::ios::binary);
  if (rotationKeyFile.is_open()) {
    if (!cc->SerializeEvalAutomorphismKey(rotationKeyFile, SerType::BINARY)) {
      std::cerr << "Error writing rotation keys" << std::endl;
      return std::make_tuple(keys, cc, -1);
    }
    std::cout << "\tRotation keys have been serialized" << std::endl;
  } else {
    std::cerr << "Error serializing Rotation keys" << std::endl;
    return std::make_tuple(keys, cc, -1);
  }

  std::ofstream sumKeyFile(DATAFOLDER + sumKeyLocation, std::ios::out | std::ios::binary);
  if (sumKeyFile.is_open()) {
    if (!cc->SerializeEvalSumKey(sumKeyFile, SerType::BINARY)) {
      std::cerr << "Error writing eval sum keys" << std::endl;
      return std::make_tuple(keys, cc, -1);
    }
    std::cout << "\tEval sum keys have been serialized" << std::endl;
  } else {
    std::cerr << "Error serializing eval sum keys" << std::endl;
    return std::make_tuple(keys, cc, -1);
  }

  if (!Serial::SerializeToFile(DATAFOLDER + cipherLocation, c1, SerType::BINARY)) {
    std::cerr << " Error writing ciphertext" << std::endl;
  }
  std::cout << "\tCiphertext have been serialized" << std::endl;

  return std::make_tuple(keys, cc, 0);
}

unsigned int connectToServer(char *szServerName, uint16_t portNum) {
    // based on https://stackoverflow.com/questions/1011339/how-do-you-make-a-http-request-with-c
    struct hostent *hp;
    unsigned int addr;
    struct sockaddr_in server;
    int conn;

    conn = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (conn == -1)
        return -1;

    if(inet_addr(szServerName)==-1)
    {
        hp=gethostbyname(szServerName);
    }
    else
    {
        addr=inet_addr(szServerName);
        hp=gethostbyaddr((char*)&addr,sizeof(addr),AF_INET);
    }

    if(hp==NULL)
    {
        close(conn);
        return -1;
    }

    server.sin_addr.s_addr=*((unsigned long*)hp->h_addr);
    server.sin_family=AF_INET;
    server.sin_port=htons(portNum);
    if(connect(conn,(struct sockaddr*)&server,sizeof(server)))
    {
        close(conn);
        return -1;
    }
    return conn;
}

unsigned int sendFileToServer(char *szServerName, uint16_t portNum, std::string file){
  unsigned int conn = connectToServer(szServerName, portNum);
  if (conn == -1){
    std::cout << "Server connection failed" << std::endl;
    return -1;
  }

  // based on https://stackoverflow.com/questions/3747086/reading-the-whole-text-file-into-a-char-array-in-c
  std::cout << "Sending file " << file << std::endl;

  long lSize;
  char *buffer;

  auto fp = fopen (file.c_str(), "rb");
  if( !fp ) {
    std::cout << "file open failed" << std::endl;
    return -1;
  }

  fseek( fp , 0L , SEEK_END);
  lSize = ftell( fp );
  rewind( fp );

  /* allocate memory for entire content */
  buffer = (char *)calloc( 1, lSize+1 );
  if( !buffer ) {
    fclose(fp);
    close(conn);
    std::cout << "memory alloc fails" << std::endl;
    return -1;
  }

  /* copy the file into the buffer */
  if( 1!=fread( buffer , lSize, 1 , fp) ){
    fclose(fp);
    free(buffer);
    close(conn);
    std::cout << "entire read fails" << std::endl;
    return -1;
  }

  std::string fileName = "";
  if (file.find("cryptocontext") != string::npos){
    fileName = "cryptocontext";
  }
  else if (file.find("key_pub") != string::npos){
    fileName = "key_pub";
  }
  else if (file.find("key_mult") != string::npos){
    fileName = "key_mult";
  }
  else if (file.find("key_rot") != string::npos){
    fileName = "key_rot";
  }
  else if (file.find("key_sum") != string::npos){
    fileName = "key_sum";
  }
  else if (file.find("ciphertext") != string::npos){
    fileName = "ciphertext";
  }

  std::string lSizeStr = std::to_string(lSize);
  std::string fileInfo = lSizeStr+"|"+fileName+"\n";

  send(conn, fileInfo.c_str(), strlen(fileInfo.c_str()), 0);
  send(conn, buffer, lSize, 0);

  /* cleanup */
  fclose(fp);
  free(buffer);
  close(conn);
  return 0;
}

unsigned int getResult(char *szServerName, uint16_t portNum, std::string reqFile){
  unsigned int conn = connectToServer(szServerName, portNum);
  if (-1 == conn){
    std::cout << "Server connection failed" << std::endl;
    return -1;
  }
  std::string resultRequest = "#";
  if (reqFile.find("cryptocontext") != string::npos){
    resultRequest = "@";
  }
  else if (reqFile.find("ciphertext") != string::npos){
    resultRequest = "#";
  }
  send(conn, resultRequest.c_str(), strlen(resultRequest.c_str()), 0);

  std::string data;
  while(1){
    char buffer[1024];
    long bytesRead = read(conn, buffer, 1);
    for (int i=0;i<bytesRead;i++){
      if ('\n' == buffer[i]) break;
      data += buffer[i];
    }
    if ((0 == bytesRead) || ('\n' == buffer[0])){
      break;
    }
  }
  long n = data.find("|");
  long fileSize = stoi(data.substr(0, n));
  std::string fileName = data.substr(n+1);

  // read file
  char* file;
  file = (char *)calloc(1, fileSize+1);
  long fileSizeRead = 0;
  while(1){
    char* buffer = (char *)calloc(1, fileSize+1);
    long bytesRead = read(conn, buffer, fileSize);
    for (int i=0;i<bytesRead;i++){
      file[i+fileSizeRead] = buffer[i];
    }
    fileSizeRead += bytesRead;
    free(buffer);
    if (0 == bytesRead){
      break;
    }
  }
  if (fileSizeRead == fileSize){
    std::cout << "file received: " << fileName << " (size:" << fileSize << ")" << std::endl;
    auto fp = fopen((FINALDATAFOLDER+"/"+fileName).c_str(), "wb");
    fwrite(file, sizeof(char), fileSize, fp);
    fclose(fp);
    free(file);
    return 0;
  }

  close(conn);
  return 0;
}

uint receiveFile(int connection, std::string &fileName_,  long &fileSize_){
  // read file info -> size|name
  std::string data;
  while(1){
    char buffer[1024];
    long bytesRead = read(connection, buffer, 1);
    for (int i=0;i<bytesRead;i++){
      if ('\n' == buffer[i]) break;
      data += buffer[i];
    }
    if ((0 == bytesRead) || ('\n' == buffer[0])){
      break;
    }
  }
  long n = data.find("|");
  long fileSize = stoi(data.substr(0, n));
  std::string fileName = data.substr(n+1);
  fileSize_ = fileSize;
  fileName_ = fileName;

  // read file
  char* file;
  file = (char *)calloc(1, fileSize+1);
  long fileSizeRead = 0;
  while(1){
    char* buffer = (char *)calloc(1, fileSize+1);
    long bytesRead = read(connection, buffer, fileSize);
    for (int i=0;i<bytesRead;i++){
      file[i+fileSizeRead] = buffer[i];
    }
    fileSizeRead += bytesRead;
    free(buffer);
    if (0 == bytesRead){
      break;
    }
  }
  if (fileSizeRead == fileSize){
    std::cout << "file received: " << fileName << " (size:" << fileSize << ")" << std::endl;
    auto fp = fopen((DATAFOLDER+"/"+fileName).c_str(), "wb");
    fwrite(file, sizeof(char), fileSize, fp);
    fclose(fp);
    free(file);
    return 0;
  }

  return -1;
}

std::tuple<Ciphertext<DCRTPoly>, uint>
deserialize(){
    // CryptoContext<DCRTPoly> cc;
    Ciphertext<DCRTPoly> c1;

    // cc->ClearEvalMultKeys();
    // cc->ClearEvalAutomorphismKeys();
    // lbcrypto::CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
    // if (!Serial::DeserializeFromFile(FINALDATAFOLDER + ccFinalLocation, cc, SerType::BINARY)) {
    //   std::cerr << "I cannot read serialized data from: " << FINALDATAFOLDER << ccFinalLocation << std::endl;
    //   return std::make_tuple(cc, c1, -1);
    // }
    // std::cout << "\tCC deserialized" << std::endl;

    if (!Serial::DeserializeFromFile(FINALDATAFOLDER + cipherFinalLocation, c1, SerType::BINARY)) {
      std::cerr << "Cannot read serialization from " << FINALDATAFOLDER + cipherFinalLocation << std::endl;
      return std::make_tuple(c1, -1);
    }
    std::cout << "\tCiphertext deserialized" << std::endl;

    return std::make_tuple(c1, 0);
}

int main() {
  std::cout << "STARTUP" << std::endl;

  // Create a socket (IPv4, TCP)
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd == -1) {
    std::cout << "Failed to create socket. errno: " << errno << std::endl;
    exit(EXIT_FAILURE);
  }

  // Listen to port 9999 on any address
  sockaddr_in sockaddr;
  sockaddr.sin_family = AF_INET;
  sockaddr.sin_addr.s_addr = INADDR_ANY;
  sockaddr.sin_port = htons(9999); // htons is necessary to convert a number to
                                   // network byte order
  if (bind(sockfd, (struct sockaddr*)&sockaddr, sizeof(sockaddr)) < 0) {
    std::cout << "Failed to bind to port 9999. errno: " << errno << std::endl;
    exit(EXIT_FAILURE);
  }

  while(1){
    // Start listening. Hold at most 10 connections in the queue
    if (listen(sockfd, 10) < 0) {
        std::cout << "Failed to listen on socket. errno: " << errno << std::endl;
        exit(EXIT_FAILURE);
    }
    // Read from the connection
    // Grab a connection from the queue
    auto addrlen = sizeof(sockaddr);
    int connection = accept(sockfd, (struct sockaddr*)&sockaddr, (socklen_t*)&addrlen);
    if (connection < 0) {
        std::cout << "Failed to grab connection. errno: " << errno << std::endl;
        exit(EXIT_FAILURE);
    }

    char buffer[32*1024];
    auto bytesRead = read(connection, buffer, 32*1024);
    // std::cout << "The message was: " << buffer;
    std::string data;
    for (auto i=0;i<bytesRead;i++){
      data += buffer[i];
    }
    std::string::size_type n = data.find("image");
    if (n != std::string::npos){
      data = data.substr(n);
      std::vector<double> image = extractImageData(data);

      auto tupleResEncrypt = ckksEncrypt(image);
      auto keys = std::get<0>(tupleResEncrypt);
      auto cc = std::get<1>(tupleResEncrypt);
      auto retCodeEncrypt = std::get<2>(tupleResEncrypt);

      sendFileToServer("localhost", 9998, DATAFOLDER+ccLocation);
      sendFileToServer("localhost", 9998, DATAFOLDER+pubKeyLocation);
      sendFileToServer("localhost", 9998, DATAFOLDER+multKeyLocation);
      sendFileToServer("localhost", 9998, DATAFOLDER+rotKeyLocation);
      sendFileToServer("localhost", 9998, DATAFOLDER+sumKeyLocation);
      sendFileToServer("localhost", 9998, DATAFOLDER+cipherLocation);

      // getResult("localhost", 9998, FINALDATAFOLDER+ccFinalLocation);
      getResult("localhost", 9998, FINALDATAFOLDER+cipherFinalLocation);

      // deserialize received data
      auto tupleResDeser = deserialize();
      auto retCodeDeser = std::get<1>(tupleResDeser);
      if (retCodeDeser != -1){
        auto cFinal = std::get<0>(tupleResDeser);
        // auto ccFinal = std::get<2>(tupleResDeser);

        Plaintext result;
        std::cout.precision(8);
        std::cout << std::endl << "Results of homomorphic computations: " << std::endl;

        cc->Decrypt(keys.secretKey, cFinal, &result);
        result->SetLength(10);
        std::cout << "xFinal = " << result;
        std::cout << "Estimated precision in bits: " << result->GetLogPrecision() << std::endl;
        auto prediction = result->GetCKKSPackedValue();
        int counter = 0;
        for (auto &value : prediction ){
          std::cout << counter << " -> " << std::real(value) << std::endl;
          counter++;
        }
      }
    }

    // Send a message to the connection
    std::string response = "-";
    send(connection, response.c_str(), response.size(), 0);
    close(connection);
  }

  // Close the connections
  close(sockfd);
}
