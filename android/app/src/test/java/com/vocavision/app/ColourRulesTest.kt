package com.vocavision.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

class ColourRulesTest {

    /** A noisy patch of one colour, like a shirt with folds and sensor noise. */
    private fun swatch(r: Int, g: Int, b: Int, count: Int = 500, seed: Long = 1, jitter: Int = 14): IntArray {
        val rnd = Random(seed)
        return IntArray(count) {
            val shade = rnd.nextInt(jitter * 2 + 1) - jitter           // shared brightness change (folds)
            val c = { v: Int -> (v + shade + rnd.nextInt(7) - 3).coerceIn(0, 255) }
            (0xFF shl 24) or (c(r) shl 16) or (c(g) shl 8) or c(b)
        }
    }

    private fun name(r: Int, g: Int, b: Int) = ColourRules.analyse(swatch(r, g, b))!!.name

    @Test fun darkGreenIsNotBlack() {
        assertEquals("dark green", name(35, 62, 48))
        assertEquals("dark green", name(30, 70, 45))
        assertEquals("dark green", name(40, 75, 55))
    }

    @Test fun lightBlueIsNotGray() {
        assertEquals("light blue", name(150, 180, 212))
        assertEquals("light blue", name(165, 190, 215))
        assertEquals("light blue", name(172, 196, 220))
    }

    @Test fun washedOutLightBlueStillBlue() {
        // Dull indoor light and auto exposure wash colour out; it must still not read as plain gray.
        assertEquals("light blue", name(168, 182, 198))
    }

    @Test fun neutrals() {
        assertEquals("white", name(235, 235, 230))
        assertEquals("white", name(215, 214, 210))
        assertEquals("black", name(25, 25, 28))
        assertEquals("black", name(38, 40, 42))
        assertEquals("gray", name(128, 128, 130))
        assertEquals("dark gray", name(80, 80, 82))
    }

    @Test fun whiteInShadeIsFlaggedForSecondOpinion() {
        val r = ColourRules.analyse(swatch(175, 175, 170))!!
        assertEquals("light gray", r.name)
        assertTrue(r.maybeWhite)
    }

    @Test fun commonShirtColours() {
        assertEquals("red", name(200, 35, 45))
        assertEquals("blue", name(40, 90, 200))
        assertEquals("navy", name(25, 32, 85))
        assertEquals("green", name(40, 150, 70))
        assertEquals("yellow", name(235, 210, 50))
        assertEquals("orange", name(240, 140, 30))
        assertEquals("pink", name(240, 160, 185))
        assertEquals("purple", name(120, 60, 160))
        assertEquals("brown", name(110, 70, 40))
        assertEquals("maroon", name(110, 25, 40))
    }

    @Test fun skinToneTrousersReadAsBeige() {
        assertEquals("beige", name(210, 165, 135))
        assertEquals("beige", name(220, 190, 150))
    }

    @Test fun majorityColourWinsOverBackground() {
        // 70% dark green shirt + 30% skin/wall behind it.
        val shirt = swatch(35, 62, 48, 350, 3)
        val other = swatch(210, 165, 135, 150, 4)
        assertEquals("dark green", ColourRules.analyse(shirt + other)!!.name)
        // 65% light blue shirt + 35% white wall.
        val blue = swatch(150, 180, 212, 325, 5)
        val wall = swatch(235, 235, 230, 175, 6)
        assertEquals("light blue", ColourRules.analyse(blue + wall)!!.name)
    }

    @Test fun redOnTheHueWrapIsNotSplit() {
        // Reds straddle 0/360 degrees; a bit of orange and pink noise must not break the vote.
        val mix = swatch(205, 30, 55, 250, 7) + swatch(210, 35, 30, 250, 8)
        assertEquals("red", ColourRules.analyse(mix)!!.name)
    }

    @Test fun emptyInputIsNull() {
        assertEquals(null, ColourRules.analyse(IntArray(0)))
        assertNotNull(ColourRules.analyse(swatch(10, 10, 10)))
    }
}
