/*
 * Created by ishaanjav
 * github.com/ishaanjav
*/

package app.ij.mlwithtensorflowlite;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;


import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import app.ij.mlwithtensorflowlite.ml.Model;


public class MainActivity extends AppCompatActivity {

    TextView result, confidence;
    ImageView imageView;
    Button picture;
    int imageSizeWidth = 224;

    int imageSize = 224;
    int imageSizeHeight = 224;
    int previewWidth = 300;
    int previewHeight = 300;
    private static final String TAG = "AndroidCameraApi";
    private Button btnTake;
    private Button btnGallery;
    private TextureView textureView;
    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();

    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 180);
        ORIENTATIONS.append(Surface.ROTATION_90, 90);
        ORIENTATIONS.append(Surface.ROTATION_180, 180);
        ORIENTATIONS.append(Surface.ROTATION_270, 270);
    }

    private String cameraId;
    protected CameraDevice cameraDevice;
    protected CameraCaptureSession cameraCaptureSessions;
    protected CaptureRequest.Builder captureRequestBuilder;
    private Size imageDimension;
    private ImageReader imageReader;
    private File file;
    private File folder;
    private String folderName = "MyPhotoDir";
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    private Handler mBackgroundHandler;
    private HandlerThread mBackgroundThread;

    private boolean isProcessingFrame = false;
    private byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;
    private Runnable postInferenceCallback;
    private Runnable imageConverter;
    private Bitmap rgbFrameBitmap;
    private SocketUtils socketUtils;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        socketUtils = new SocketUtils();
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        new Thread(new Runnable() {
            @Override
            public void run() {
                socketUtils.createSocket();
                while (true) {
                    try {
                        try {
                            if (socketUtils.isSocketAvailable()) {
                                String isHas = socketUtils.sendMessage("1");
                                if (isHas != null) {
                                    if (isHas.equals("INC")) {
                                        runOnUiThread(new Runnable() {
                                            @Override
                                            public void run() {
                                                picture.callOnClick();
                                            }
                                        });
                                    }
                                }
                            } else {
                                socketUtils.createSocket();
                            }
                        } catch (Exception e) {
                            Log.e("SocketUtils", "error: "+e.getMessage());
                            socketUtils = new SocketUtils();
                            socketUtils.createSocket();
                        }
                        Thread.sleep(500);
                    } catch (Exception e) {
                        Log.e("SocketUtils", "error");
                    }
                }
            }
        }).start();

        textureView = findViewById(R.id.texture);
        imageReader = ImageReader.newInstance(previewWidth, previewHeight, ImageFormat.YUV_420_888, 1);
        imageReader.setOnImageAvailableListener(new ImageReader.OnImageAvailableListener() {
            @Override
            public void onImageAvailable(ImageReader imageReader) {
                if (previewWidth == 0 || previewHeight == 0) {
                    return;
                }
                if (rgbBytes == null) {
                    rgbBytes = new int[previewWidth * previewHeight];
                }
                try {
                    final Image image = imageReader.acquireLatestImage();

                    if (image == null) {
                        Log.e("PIZDA NET IMAGI", "PIZDEC");
                        return;
                    }

                    if (isProcessingFrame) {
                        image.close();
                        return;
                    }
                    isProcessingFrame = true;
                    final Image.Plane[] planes = image.getPlanes();
                    fillBytes(planes, yuvBytes);
                    yRowStride = planes[0].getRowStride();
                    final int uvRowStride = planes[1].getRowStride();
                    final int uvPixelStride = planes[1].getPixelStride();
                    Log.e("VSE ZAEBATO", "VSE ZAEBATO");
                    imageConverter =
                            new Runnable() {
                                @Override
                                public void run() {
                                    try {
                                        ImageUtils.convertYUV420ToARGB8888(
                                                yuvBytes[0],
                                                yuvBytes[1],
                                                yuvBytes[2],
                                                previewWidth,
                                                previewHeight,
                                                yRowStride,
                                                uvRowStride,
                                                uvPixelStride,
                                                rgbBytes);
                                    } catch (Exception e) {
                                        Log.e("IN UMAGE", e.getMessage());
                                    }

                                }
                            };
                    Log.d("MSG", "PROSHEL CONVERT");
                    postInferenceCallback =
                            new Runnable() {
                                @Override
                                public void run() {
                                    image.close();
                                    isProcessingFrame = false;
                                }
                            };
                    image.close();
                    processImage();


                } catch (final Exception e) {
                    e.printStackTrace();
                    //Log.d("tryError", );
                    return;
                }
            }
        }, null);
        if(textureView != null)
            textureView.setSurfaceTextureListener(textureListener);

        if(picture != null)
            picture.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    takePicture();
                }
            });

//        picture.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                // Launch camera if we have permission
//                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
//                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
//                    startActivityForResult(cameraIntent, 1);
//                } else {
//                    //Request camera permission if we don't have it.
//                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
//                }
//            }
//        });
    }

//    public void classifyImage(Bitmap image) {
//        try {
//            Model model = Model.newInstance(getApplicationContext());
//
//            // Creates inputs for reference.
//            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
//            float imw = imageSizeWidth;
//            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(imageSizeWidth * imageSizeHeight * 3);
//            byteBuffer.order(ByteOrder.nativeOrder());
////            TensorImage tensorImage = new TensorImage(DataType.UINT8);
////            tensorImage.load(image);
//
//            // get 1D array of 224 * 224 pixels in image
//            int [] intValues = new int[imageSizeWidth * imageSizeHeight];
//            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
////
////            // iterate over pixels and extract R, G, and B values. Add to bytebuffer.
//            int pixel = 0;
//            for(int i = 0; i < imageSizeWidth; i++){
//                for(int j = 0; j < imageSizeHeight; j++){
//                    int val = intValues[pixel++]; // RGB
//                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
//                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
//                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
//                }
//            }
//
//            inputFeature0.loadBuffer(byteBuffer);
//
//            // Runs model inference and gets result.
////            Model.Outputs outputs = model.process(inputFeature0);
////            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
////            Log.d("test", ""+outputFeature0.getDataType());
////            TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();
////            TensorBuffer outputFeature2 = outputs.getOutputFeature2AsTensorBuffer();
////            TensorBuffer outputFeature3 = outputs.getOutputFeature3AsTensorBuffer();
////            TensorBuffer outputFeature4 = outputs.getOutputFeature4AsTensorBuffer();
////            TensorBuffer outputFeature5 = outputs.getOutputFeature5AsTensorBuffer();
//
//            // Releases model resources if no longer used.
//            model.close();
//        } catch (Exception e) {
//            Log.e("tryError in proc", ""+e.getMessage());
//            // TODO Handle the exception
//        }
//    }


    private void processImage() {
        imageConverter.run();
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
        //Do your work here
        Bitmap image = Bitmap.createBitmap(rgbFrameBitmap,10,10,270,270);
        Log.e("TESTIMIMAGE", "TESTIMIMAGE");
        int dimension = Math.min(image.getWidth(), image.getHeight());
        image = ThumbnailUtils.extractThumbnail(image, imageSizeWidth, imageSizeHeight);
        imageView.setImageBitmap(image);

        image = Bitmap.createScaledBitmap(image, imageSizeWidth, imageSizeHeight, false);
        classifyImage(image);
        imageView.setImageBitmap(rgbFrameBitmap);

        postInferenceCallback.run();
    }

    protected void fillBytes(final Image.Plane[] planes, final byte[][] yuvBytes) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    TextureView.SurfaceTextureListener textureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            // Open your camera here
            openCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
            // Transform you image captured size according to the surface width and height
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            return false;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        }
    };

    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
            // This is called when the camera is open
            Log.e(TAG, "onOpened");
            cameraDevice = camera;
            createCameraPreview();
        }

        @Override
        public void onDisconnected(CameraDevice camera) {
            cameraDevice.close();
        }

        @Override
        public void onError(CameraDevice camera, int error) {
            cameraDevice.close();
            cameraDevice = null;
        }
    };

    protected void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("Camera Background");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    protected void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    protected void takePicture() {
        try {

            CaptureRequest.Builder captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);
            captureRequest.addTarget(imageReader.getSurface());
            int rotation = getWindowManager().getDefaultDisplay().getRotation();
            captureRequest.set(CaptureRequest.JPEG_ORIENTATION, ORIENTATIONS.get(rotation));
            Log.d("ROT", ""+rotation);
            CameraCaptureSession.CaptureCallback captureCallback = new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureCompleted(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull TotalCaptureResult result) {
                    super.onCaptureCompleted(session, request, result);

                    Toast.makeText(getApplicationContext(), "VSE TOP!"+request.get(CaptureRequest.JPEG_ORIENTATION), Toast.LENGTH_SHORT).show();
                }
            };
            cameraCaptureSessions.capture(captureRequest.build(), captureCallback, null);

        } catch (CameraAccessException e) {
            throw new RuntimeException(e);
        }

    }

    protected void createCameraPreview() {
        try {
            SurfaceTexture texture = textureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(imageDimension.getWidth(), imageDimension.getHeight());
            Surface surface = new Surface(texture);
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface);
            cameraDevice.createCaptureSession(Arrays.asList(surface, imageReader.getSurface()), new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                    //The camera is already closed
                    if (null == cameraDevice) {
                        return;
                    }
                    // When the session is ready, we start displaying the preview.
                    cameraCaptureSessions = cameraCaptureSession;
                    updatePreview();
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                    Toast.makeText(MainActivity.this, "Configuration change", Toast.LENGTH_SHORT).show();
                }
            }, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void openCamera() {
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        Log.e(TAG, "is camera open");
        try {
            cameraId = manager.getCameraIdList()[0];
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            assert map != null;
            imageDimension = map.getOutputSizes(SurfaceTexture.class)[0];
            // Add permission for camera and let user grant the permission
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
                return;
            }
            manager.openCamera(cameraId, stateCallback, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
        Log.e(TAG, "openCamera X");
    }

    protected void updatePreview() {
        if (null == cameraDevice) {
            Log.e(TAG, "updatePreview error, return");
        }
        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
        try {
            cameraCaptureSessions.setRepeatingRequest(captureRequestBuilder.build(), null, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                // close the app
                Toast.makeText(MainActivity.this, "Sorry!!!, you can't use this app without granting permission", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.e(TAG, "onResume");
        startBackgroundThread();
        if (textureView.isAvailable()) {
            openCamera();
        } else {
            textureView.setSurfaceTextureListener(textureListener);
        }
    }

    @Override
    protected void onPause() {
        Log.e(TAG, "onPause");
        cameraDevice.close();
        stopBackgroundThread();
        super.onPause();
    }


    String s = "";




    public void classifyImage(Bitmap image){
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // get 1D array of 224 * 224 pixels in image
            int [] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            // iterate over pixels and extract R, G, and B values. Add to bytebuffer.
            int pixel = 0;
            for(int i = 0; i < imageSize; i++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for(int i = 0; i < confidences.length; i++){
                if(confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Бумага", "Пластик", "Стекло", "Металл"};
            result.setText(classes[maxPos]);

            s = "";
            float maxConfidences = 0;
            for(int i = 0; i < 4; i++){
                if(confidences[i] >= 0) { // Проверяем вероятность
                    if (confidences[i] > maxConfidences) {
                        s += String.format("%s: %.1f%%\n", classes[i], confidences[i] * 100);
                        maxConfidences = confidences[i];
                    }
                }
            }
            confidence.setText(s);
            new Thread(new Runnable() {
                @Override
                public void run() {
                    if(s.isEmpty()) {
                        socketUtils.sendMessage("nA");
                    } else {
                        String[] split = s.split(":");
                        socketUtils.sendMessage(result.getText().toString());
                    }


                }
            }).start();
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }


    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);

//            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
//            classifyImage(image);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}