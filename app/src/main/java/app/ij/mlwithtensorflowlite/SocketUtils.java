package app.ij.mlwithtensorflowlite;

import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.Socket;

public class SocketUtils {
    Socket clientSocket; // этой строкой мы запрашиваем
    PrintWriter printWriter;
    BufferedReader in;
    public void createSocket() {
        try {
            clientSocket = new Socket("192.168.1.1", 1111); // этой строкой мы запрашиваем
            OutputStream outputStream = clientSocket.getOutputStream();
            printWriter = new PrintWriter(outputStream, true);
            InputStream inputStream = clientSocket.getInputStream();
            in = new BufferedReader(new InputStreamReader(inputStream));
            Log.d("SOCKET", "Socket Created!");
        } catch (Exception e) {
            Log.e("SocketUtils", "Something went wrong while socket creating: "+e.getMessage());
        }
    }
    public String sendMessage(String word) {
        printWriter.println(word);
        printWriter.flush();
        try {
            //Log.d("MSG", "SENDED MESSAGE");
            return in.readLine();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public boolean isSocketAvailable(){
        return clientSocket.isConnected();
    }
}
