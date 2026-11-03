package com.vocavision.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Capture-then-analyse flow: the camera only shows a preview. One tap takes a still photo,
 * which is analysed once at full quality and spoken once, so a shaky camera cannot garble the voice.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var preview: PreviewView
    private lateinit var photoView: ImageView
    private lateinit var overlay: OverlayView
    private lateinit var headline: TextView
    private lateinit var detail: TextView
    private lateinit var timeText: TextView
    private lateinit var captureBtn: Button
    private lateinit var speakBtn: Button
    private lateinit var repeatBtn: Button
    private lateinit var flipBtn: Button
    private lateinit var galleryBtn: Button

    private val worker = Executors.newSingleThreadExecutor()
    @Volatile private var pipeline: Pipeline? = null
    @Volatile private var busy = false

    private var imageCapture: ImageCapture? = null
    private var useFront = false
    private var showingResult = false
    private var lastSentence = ""

    private var speakOn = true
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    private val askCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera() else headline.text = "Camera permission is needed to take photos."
    }
    private val pickPhoto = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) analyseUri(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        preview = findViewById(R.id.preview)
        photoView = findViewById(R.id.photo)
        overlay = findViewById(R.id.overlay)
        headline = findViewById(R.id.headline)
        detail = findViewById(R.id.detail)
        timeText = findViewById(R.id.fps)
        captureBtn = findViewById(R.id.btn_capture)
        speakBtn = findViewById(R.id.btn_speak)
        repeatBtn = findViewById(R.id.btn_repeat)
        flipBtn = findViewById(R.id.btn_flip)
        galleryBtn = findViewById(R.id.btn_photo)

        timeText.text = "Ready"
        headline.text = "Loading the AI model..."
        captureBtn.isEnabled = false

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val t = tts
                if (t != null) {
                    val locale = Locale.getDefault()
                    t.language = if (t.isLanguageAvailable(locale) >= TextToSpeech.LANG_AVAILABLE) locale else Locale.US
                }
                ttsReady = true
            }
        }

        captureBtn.setOnClickListener { if (showingResult) retake() else capture() }
        speakBtn.setOnClickListener {
            speakOn = !speakOn
            speakBtn.text = if (speakOn) "Voice on" else "Voice off"
            if (!speakOn) tts?.stop()
        }
        repeatBtn.setOnClickListener { if (lastSentence.isNotEmpty()) speak(lastSentence, force = true) }
        flipBtn.setOnClickListener {
            if (showingResult) retake()
            useFront = !useFront
            startCamera()
        }
        galleryBtn.setOnClickListener { pickPhoto.launch("image/*") }

        worker.execute {
            try {
                val yolo = Detector(applicationContext)
                // CLIP is the safety net for photos where YOLO is unsure; the app still works without it.
                var clipError: String? = null
                val clip = try { ClipClassifier(applicationContext) } catch (e: Throwable) { clipError = e.message; null }
                pipeline = Pipeline(yolo, clip)
                runOnUiThread {
                    headline.text = "Point the camera at clothing or food, then tap Capture"
                    if (clipError != null) detail.text = "Warning: CLIP failed to load, so accuracy is reduced. $clipError"
                    captureBtn.isEnabled = true
                }
            } catch (e: Throwable) {
                runOnUiThread { headline.text = "Could not load the model: ${e.message}" }
            }
        }

        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else askCamera.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val selector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .build()

            val previewCase = Preview.Builder().setResolutionSelector(selector).build()
            previewCase.setSurfaceProvider(preview.surfaceProvider)

            val capture = ImageCapture.Builder()
                .setResolutionSelector(selector)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()
            imageCapture = capture

            val camera = if (useFront) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, camera, previewCase, capture)
            } catch (e: Exception) {
                Toast.makeText(this, "Camera not available: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun capture() {
        val cap = imageCapture
        if (cap == null || busy || pipeline == null) return
        busy = true
        captureBtn.isEnabled = false
        headline.text = "Analysing..."
        detail.text = ""
        cap.takePicture(worker, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val bmp = try {
                    val rotation = image.imageInfo.rotationDegrees
                    val raw = image.toBitmap()
                    shrink(rotate(raw, rotation.toFloat()))
                } catch (e: Throwable) {
                    null
                } finally {
                    image.close()
                }
                if (bmp == null) fail("Could not read the photo. Try again.") else analyse(bmp)
            }

            override fun onError(exception: ImageCaptureException) {
                fail("Capture failed: ${exception.message}")
            }
        })
    }

    private fun analyseUri(uri: Uri) {
        if (busy || pipeline == null) return
        busy = true
        captureBtn.isEnabled = false
        headline.text = "Analysing..."
        detail.text = ""
        worker.execute {
            val bmp = loadUpright(uri)
            if (bmp == null) fail("Could not open that photo.") else analyse(bmp)
        }
    }

    /** Runs on the worker thread: detect once, then show and speak the result. */
    private fun analyse(bmp: Bitmap) {
        try {
            val t0 = SystemClock.elapsedRealtime()
            val result = pipeline!!.run(bmp)
            val ms = SystemClock.elapsedRealtime() - t0
            runOnUiThread {
                photoView.setImageBitmap(bmp)
                photoView.visibility = View.VISIBLE
                preview.visibility = View.INVISIBLE
                overlay.show(result.found, bmp.width.toFloat() / bmp.height, false)
                headline.text = result.sentence
                detail.text = result.found.joinToString("  •  ") { "${it.label} ${(it.det.conf * 100).toInt()}%" } +
                    (if (result.note.isEmpty()) "" else "\n" + result.note)
                timeText.text = "$ms ms"
                lastSentence = result.sentence
                showingResult = true
                busy = false
                captureBtn.text = "Retake"
                captureBtn.isEnabled = true
                speak(result.sentence, force = true)
            }
        } catch (e: Throwable) {
            fail("Error: ${e.message}")
        }
    }

    private fun fail(message: String) {
        runOnUiThread {
            busy = false
            captureBtn.isEnabled = pipeline != null
            headline.text = message
        }
    }

    private fun retake() {
        tts?.stop()
        showingResult = false
        photoView.visibility = View.GONE
        preview.visibility = View.VISIBLE
        overlay.show(emptyList(), 3f / 4f, false)
        headline.text = "Point the camera at clothing or food, then tap Capture"
        detail.text = ""
        timeText.text = "Ready"
        captureBtn.text = "Capture and analyse"
    }

    private fun speak(text: String, force: Boolean) {
        if ((speakOn || force) && ttsReady) tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "vv")
    }

    private fun rotate(src: Bitmap, degrees: Float): Bitmap =
        if (degrees == 0f) src
        else Bitmap.createBitmap(src, 0, 0, src.width, src.height, Matrix().apply { postRotate(degrees) }, true)

    /** Photos from modern phones are 12 MP+; 1600 px on the long side is plenty and keeps it quick. */
    private fun shrink(src: Bitmap): Bitmap {
        val longSide = maxOf(src.width, src.height)
        if (longSide <= MAX_SIDE) return src
        val s = MAX_SIDE.toFloat() / longSide
        return Bitmap.createScaledBitmap(src, (src.width * s).toInt(), (src.height * s).toInt(), true)
    }

    /** Decodes a picked image at a sensible size and applies its EXIF rotation. */
    private fun loadUpright(uri: Uri): Bitmap? = try {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > MAX_SIDE) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        val decoded = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
        val orientation = contentResolver.openInputStream(uri)?.use {
            ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        } ?: ExifInterface.ORIENTATION_NORMAL
        val degrees = when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        decoded?.let { rotate(it, degrees) }
    } catch (e: Exception) {
        null
    }

    override fun onDestroy() {
        super.onDestroy()
        tts?.shutdown()
        worker.shutdown()
    }

    companion object {
        private const val MAX_SIDE = 1600
    }
}
