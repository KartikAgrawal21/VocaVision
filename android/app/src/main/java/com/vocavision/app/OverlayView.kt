package com.vocavision.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/** Draws detection boxes over the camera preview or a picked photo (both are fit-centred). */
class OverlayView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    private var found: List<Found> = emptyList()
    private var aspect = 3f / 4f   // upright image width / height
    private var mirror = false

    private val density = resources.displayMetrics.density
    private val box = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f * density
    }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 14f * density
        isFakeBoldText = true
    }

    fun show(items: List<Found>, imageAspect: Float, mirrored: Boolean) {
        found = items
        aspect = imageAspect
        mirror = mirrored
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        if (found.isEmpty()) return
        val vw = width.toFloat()
        val vh = height.toFloat()
        val iw: Float
        val ih: Float
        if (vw / vh > aspect) { ih = vh; iw = vh * aspect } else { iw = vw; ih = vw / aspect }
        val ox = (vw - iw) / 2f
        val oy = (vh - ih) / 2f

        for (f in found) {
            val d = f.det
            val a = if (mirror) 1f - d.x2 else d.x1
            val b = if (mirror) 1f - d.x1 else d.x2
            val r = RectF(ox + a * iw, oy + d.y1 * ih, ox + b * iw, oy + d.y2 * ih)
            val colour = if (f.isFood) FOOD_COLOUR else CLOTHING_COLOUR
            box.color = colour
            canvas.drawRoundRect(r, 8f * density, 8f * density, box)

            val label = "${f.label} ${(d.conf * 100).toInt()}%"
            val pad = 6f * density
            val tw = text.measureText(label)
            val th = text.textSize
            val top = (r.top - th - pad * 2).coerceAtLeast(oy)
            fill.color = colour
            canvas.drawRoundRect(RectF(r.left, top, r.left + tw + pad * 2, top + th + pad * 2), 6f * density, 6f * density, fill)
            canvas.drawText(label, r.left + pad, top + th + pad * 0.6f, text)
        }
    }

    companion object {
        private val FOOD_COLOUR = Color.rgb(255, 152, 0)
        private val CLOTHING_COLOUR = Color.rgb(46, 125, 255)
    }
}
