package com.vocavision.app

import android.graphics.Bitmap

/** A detection plus the human label shown on screen and spoken aloud. */
class Found(val det: Det, val label: String, val isFood: Boolean)

class Description(val sentence: String, val found: List<Found>, val note: String = "") {
    val isEmpty get() = found.isEmpty()
}

/**
 * Combines YOLO and CLIP the same way the desktop vision_engine.py does:
 *  - YOLO finds nothing           -> CLIP names the main item
 *  - YOLO finds one weak/odd item -> CLIP double-checks it and may override
 *  - YOLO finds several things    -> keep YOLO's multi-object result (shirt and trousers)
 */
class Pipeline(private val detector: Detector, private val clip: ClipClassifier?) {

    private val notes = ArrayList<String>()

    fun run(frame: Bitmap): Description {
        notes.clear()
        val d = runInternal(frame)
        return Description(d.sentence, d.found, notes.joinToString("\n"))
    }

    private fun runInternal(frame: Bitmap): Description {
        val dets = detector.detect(frame, YOLO_SIZE)

        if (dets.isEmpty()) {
            return if (clip == null) Describer.sentence(emptyList()) else fromClip(frame, clip.classify(frame))
        }

        if (clip != null) {
            val best = dets.maxByOrNull { it.conf }!!
            val strong = dets.filter { it.conf >= 0.75f }.map { it.cls }.toSet()
            if (strong.size < 2) {
                val guess = clip.classify(frame)
                val a = norm(guess.name)
                val b = norm(Labels.rawName(best.cls))
                val agrees = a in b || b in a
                val clipWins = guess.kind != "none" && !agrees &&
                    (guess.conf >= 0.85f || (best.conf < 0.60f && guess.conf >= best.conf))
                if (clipWins) return fromClip(frame, guess)
            }
        }
        return fromYolo(frame, dets)
    }

    private fun fromYolo(frame: Bitmap, dets: List<Det>): Description {
        val found = dets.map { d ->
            val food = Labels.isFood(d.cls)
            val name = Labels.display(d.cls)
            val colour = if (food) null else colourOfBox(frame, d)
            Found(d, if (colour != null) "$colour $name" else name, food)
        }
        return Describer.sentence(found)
    }

    private fun colourOfBox(frame: Bitmap, d: Det): String? {
        val w = d.x2 - d.x1
        val h = d.y2 - d.y1
        if (w > 0.05f && h > 0.05f) {
            return colourFor(frame, d.x1 + w * 0.2f, d.y1 + h * 0.2f, d.x2 - w * 0.2f, d.y2 - h * 0.2f)
        }
        return ColorNamer.name(frame, d.x1, d.y1, d.x2, d.y2)
    }

    /**
     * Colour of a region, measured from its pixels (hue first, then dark/light). CLIP is only asked
     * when the pixels say "light gray", which in dim light can really be a white item.
     */
    private fun colourFor(frame: Bitmap, l: Float, t: Float, r: Float, b: Float): String? {
        val px = ColorNamer.region(frame, l, t, r, b)
        if (px == null) {
            return try { clip?.colourOf(frame, l, t, r, b) } catch (_: Throwable) { null }
        }
        var name = px.name
        var note = "pixels=${px.name} (${(px.share * 100).toInt()}% of area, S=${"%.2f".format(px.saturation)}, V=${"%.2f".format(px.value)})"
        if (px.maybeWhite && clip != null) {
            val cl = try { clip.colourOf(frame, l, t, r, b) } catch (_: Throwable) { null }
            note += " clip=$cl"
            if (cl == "white") name = "white"
        }
        notes.add("colour: $note -> $name")
        return name
    }

    /** Builds the result from CLIP alone: the top item, plus shirt/trousers if both are clearly present. */
    private fun fromClip(frame: Bitmap, g: ClipClassifier.Guess): Description {
        val c = clip!!
        if (g.kind == "none" || g.conf < 0.30f) {
            return Description("I am not sure what this is. Please move closer and try again.", emptyList())
        }
        val topCls = Labels.fromClip(g.name)
        if (topCls < 0) return Description("I am not sure what this is. Please move closer and try again.", emptyList())

        // A person in shirt and trousers: report both when CLIP also rates the second garment.
        var upperCls = -1
        var lowerCls = -1
        if (Labels.isUpperBody(topCls)) {
            upperCls = topCls
            if (probOf(c, g, Labels.TROUSER) >= SECOND_PIECE_MIN) lowerCls = Labels.TROUSER
        } else if (topCls == Labels.TROUSER) {
            lowerCls = topCls
            upperCls = findBestUpper(c, g)
        }
        val both = upperCls >= 0 && lowerCls >= 0

        val items = ArrayList<Triple<Int, Float, FloatArray>>()  // class, confidence, box
        if (both) {
            items.add(Triple(upperCls, probOf(c, g, upperCls), floatArrayOf(0.05f, 0.05f, 0.95f, 0.55f)))
            items.add(Triple(lowerCls, probOf(c, g, lowerCls), floatArrayOf(0.05f, 0.45f, 0.95f, 0.97f)))
        } else {
            items.add(Triple(topCls, g.conf, floatArrayOf(0.05f, 0.05f, 0.95f, 0.95f)))
        }

        val found = items.map { (cls, p, box) ->
            val food = Labels.isFood(cls)
            val name = Labels.display(cls)
            val label = if (food) name else {
                val colour = try {
                    when {
                        both && Labels.isUpperBody(cls) -> colourFor(frame, 0.25f, 0.15f, 0.75f, 0.5f)
                        both -> colourFor(frame, 0.3f, 0.6f, 0.7f, 0.95f)
                        Labels.isUpperBody(cls) -> colourFor(frame, 0.28f, 0.6f, 0.72f, 0.98f)  // desktop's tuned crop
                        else -> colourFor(frame, 0.3f, 0.3f, 0.7f, 0.7f)
                    }
                } catch (_: Throwable) { null }
                if (colour != null) "$colour $name" else name
            }
            Found(Det(cls, p, box[0], box[1], box[2], box[3]), label, food)
        }
        return Describer.sentence(found)
    }

    private fun findBestUpper(c: ClipClassifier, g: ClipClassifier.Guess): Int {
        var best = -1
        var bestP = SECOND_PIECE_MIN
        for (i in g.probs.indices) {
            val cls = Labels.fromClip(c.name(i))
            if (cls >= 0 && Labels.isUpperBody(cls) && g.probs[i] >= bestP) { best = cls; bestP = g.probs[i] }
        }
        return best
    }

    private fun probOf(c: ClipClassifier, g: ClipClassifier.Guess, cls: Int): Float {
        for (i in g.probs.indices) if (Labels.fromClip(c.name(i)) == cls) return g.probs[i]
        return 0f
    }

    private fun norm(s: String) = s.lowercase().filter { it.isLetter() }

    companion object {
        private const val YOLO_SIZE = 640
        private const val SECOND_PIECE_MIN = 0.12f
    }
}

/** Builds the spoken sentence, mirroring _build_description in the desktop vision_engine.py. */
object Describer {

    fun sentence(found: List<Found>): Description {
        val foods = found.filter { it.isFood }
        val clothes = found.filter { !it.isFood }

        val parts = ArrayList<String>()
        if (foods.isNotEmpty()) parts.add("I can see ${oxford(counted(foods))}.")
        if (clothes.isNotEmpty()) {
            val lead = if (parts.isEmpty()) "I can see" else "I can also see"
            parts.add("$lead ${oxford(counted(clothes))}.")
        }
        val text = if (parts.isEmpty()) "I could not detect any known objects." else parts.joinToString(" ")
        return Description(text, found)
    }

    private fun counted(items: List<Found>): List<String> =
        items.groupingBy { it.label }.eachCount().map { (label, n) ->
            if (n == 1) "a $label" else "$n $label${if (label.endsWith("s")) "" else "s"}"
        }

    private fun oxford(items: List<String>): String = when (items.size) {
        1 -> items[0]
        2 -> "${items[0]} and ${items[1]}"
        else -> items.dropLast(1).joinToString(", ") + ", and " + items.last()
    }
}
