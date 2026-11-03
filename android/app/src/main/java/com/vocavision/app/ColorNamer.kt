package com.vocavision.app

import android.graphics.Bitmap

/** Samples a region of a photo and hands the pixels to [ColourRules]. */
object ColorNamer {

    private const val GRID = 24

    /** Colour of a region given as fractions of the image (left, top, right, bottom). */
    fun region(src: Bitmap, l: Float, t: Float, r: Float, b: Float): ColourRules.Result? {
        val x = (l * src.width).toInt().coerceIn(0, src.width - 2)
        val y = (t * src.height).toInt().coerceIn(0, src.height - 2)
        val w = ((r - l) * src.width).toInt().coerceIn(2, src.width - x)
        val h = ((b - t) * src.height).toInt().coerceIn(2, src.height - y)
        if (w < 8 || h < 8) return null

        val small = Bitmap.createScaledBitmap(Bitmap.createBitmap(src, x, y, w, h), GRID, GRID, true)
        val px = IntArray(GRID * GRID)
        small.getPixels(px, 0, GRID, 0, 0, GRID, GRID)
        small.recycle()
        return ColourRules.analyse(px)
    }

    /** Middle 60% of a box (less background around the garment). */
    fun name(src: Bitmap, x1: Float, y1: Float, x2: Float, y2: Float): String? {
        val w = x2 - x1
        val h = y2 - y1
        return region(src, x1 + w * 0.2f, y1 + h * 0.2f, x2 - w * 0.2f, y2 - h * 0.2f)?.name
    }
}
