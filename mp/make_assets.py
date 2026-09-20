"""Generate every texture the multiplayer game adds on top of Crafter's own (16x16 RGBA, same style).

    python -m mp.make_assets

  buckets        wood / iron, empty and full
  campfire       wood and stone-ringed, each lit / dying down / burnt out
  players        one shirt colour per player slot (matches the GUI), all four facings + asleep + downed
  dropped items  the item icon, smaller, sitting on a shadow with a sparkle -- so a dropped stone does
                 not look like a stone block
  shade          the night-only monster
  numbers        10, 11, 12 for the iron bucket's water count (Crafter only ships digits 1-9)
"""
import colorsys
import pathlib

import imageio.v3 as imageio
import numpy as np
from crafter import constants
from PIL import Image, ImageDraw

OUT = pathlib.Path(__file__).parent / "assets"
SRC = constants.root / "assets"
PLAYER_HUES = [0.60, 0.05, 0.38, 0.14, 0.92, 0.75]      # blue, orange, green, yellow, pink, violet
DROPPABLE = ["sapling", "wood", "stone", "coal", "iron", "diamond", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
             "wood_sword", "stone_sword", "iron_sword"]


def load(name):
    a = imageio.imread((SRC / (name + ".png")).read_bytes())
    if a.shape[-1] == 3:
        a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
    return a


def save(name, arr_or_img):
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    img.save(OUT / (name + ".png"))


def bucket(body, rim, water=None):
    im = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.arc([3, 0, 12, 9], 180, 360, fill=(60, 60, 60, 255))
    d.polygon([(2, 5), (13, 5), (11, 14), (4, 14)], fill=body)
    d.line([(2, 5), (13, 5)], fill=rim)
    d.line([(4, 14), (11, 14)], fill=rim)
    d.line([(3, 9), (12, 9)], fill=rim)
    if water:
        d.rectangle([3, 6, 12, 7], fill=water)
    return im


def campfire(state, stone=False):
    """state: 'lit', 'low' (about to go out: a small flame) or 'out'. A stone campfire sits in a ring of stones."""
    im = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if stone:
        for x, y in ((1, 12), (4, 14), (8, 15), (12, 14), (14, 12), (2, 9), (13, 9)):
            d.rectangle([x, y - 1, x + 1, y], fill=(120, 120, 128, 255))
            d.point([(x, y - 1)], fill=(170, 170, 178, 255))
    d.line([(3, 13), (12, 10)], fill=(92, 58, 28, 255), width=2)
    d.line([(3, 10), (12, 13)], fill=(122, 78, 36, 255), width=2)
    if state == "lit":
        d.polygon([(8, 1), (12, 8), (10, 12), (6, 12), (4, 8)], fill=(235, 104, 52, 255))
        d.polygon([(8, 5), (10, 9), (8, 12), (6, 9)], fill=(250, 210, 60, 255))
    elif state == "low":
        d.polygon([(8, 7), (10, 10), (8, 12), (6, 10)], fill=(235, 104, 52, 255))
        d.point([(8, 10), (8, 11)], fill=(250, 210, 60, 255))
    else:
        d.ellipse([5, 8, 11, 12], fill=(70, 70, 70, 255))
    return im


def shade():
    """A night monster: a ragged shadow with two pale eyes."""
    im = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([3, 1, 12, 11], fill=(28, 16, 48, 235))
    d.polygon([(3, 7), (12, 7), (14, 15), (11, 12), (9, 15), (7, 12), (5, 15), (2, 13)], fill=(28, 16, 48, 235))
    d.ellipse([2, 4, 13, 12], outline=(72, 40, 120, 160))
    d.rectangle([5, 5, 6, 6], fill=(214, 190, 255, 255))
    d.rectangle([9, 5, 10, 6], fill=(214, 190, 255, 255))
    return im


def recolour_shirt(sprite, hue):
    """Crafter's player wears teal (hue 0.52). Move just those pixels to `hue`, keeping shade and saturation."""
    out = sprite.copy()
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            r, g, b, a = (int(v) for v in out[y, x])
            if a == 0:
                continue
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if 0.45 <= h <= 0.60 and s > 0.25:
                out[y, x, :3] = [round(c * 255) for c in colorsys.hsv_to_rgb(hue, s, v)]
    return out


def downed(sprite):
    out = sprite.astype(np.float32)
    solid = out[..., 3] > 0
    out[solid, :3] = 0.5 * out[solid, :3] + 0.5 * np.array([208, 59, 59])        # bled-out red wash
    img = Image.fromarray(out.astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle([11, 1, 13, 5], fill=(255, 255, 255, 255))                           # white cross: "revive me"
    d.rectangle([10, 2, 14, 4], fill=(255, 255, 255, 255))
    d.line([(12, 2), (12, 4)], fill=(208, 59, 59, 255))
    d.line([(11, 3), (13, 3)], fill=(208, 59, 59, 255))
    return img


def dropped(icon):
    im = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([2, 11, 13, 15], fill=(0, 0, 0, 110))                 # ground shadow
    small = Image.fromarray(icon).resize((11, 11), Image.NEAREST)
    im.alpha_composite(small, (2, 2))
    d = ImageDraw.Draw(im)
    d.point([(13, 1), (13, 3), (12, 2), (14, 2)], fill=(255, 240, 120, 255))         # sparkle: can be picked up
    d.point([(13, 2)], fill=(255, 255, 255, 255))
    return im


def zero():
    """Crafter ships digits 1-9 only. A zero in the same style: white glyph, black outline."""
    im = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([3, 2, 12, 14], radius=3, fill=(0, 0, 0, 255))
    d.rounded_rectangle([4, 3, 11, 13], radius=2, fill=(255, 255, 255, 255))
    d.rectangle([6, 5, 9, 11], fill=(0, 0, 0, 255))
    return np.array(im)


def two_digits(n):
    im = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for i, ch in enumerate(str(n)):
        digit = Image.fromarray(zero() if ch == "0" else load(ch)).resize((9, 16), Image.NEAREST)
        im.alpha_composite(digit, (i * 7, 0))
    return im


def main():
    OUT.mkdir(exist_ok=True)
    wood, wood_rim, iron, iron_rim, water = (139, 94, 52, 255), (92, 58, 28, 255), (176, 180, 188, 255), (110, 114, 122, 255), (64, 140, 230, 255)
    save("wood_bucket", bucket(wood, wood_rim))
    save("wood_bucket_full", bucket(wood, wood_rim, water))
    save("iron_bucket", bucket(iron, iron_rim))
    save("iron_bucket_full", bucket(iron, iron_rim, water))
    for stone in (False, True):
        for state, suffix in (("lit", ""), ("low", "_low"), ("out", "_out")):
            save(("stone_" if stone else "") + "campfire" + suffix, campfire(state, stone))
    for pid, hue in enumerate(PLAYER_HUES):
        for pose in ("left", "right", "up", "down", "sleep"):
            save("p%d-player-%s" % (pid, pose), recolour_shirt(load("player-" + pose), hue))
        save("p%d-player-downed" % pid, downed(recolour_shirt(load("player-sleep"), hue)))
    for kind in DROPPABLE:
        save("dropped_" + kind, dropped(load(kind)))
    save("shade", shade())
    for n in (10, 11, 12):
        save(str(n), two_digits(n))
    print("wrote %d textures to %s" % (len(list(OUT.glob("*.png"))), OUT))


if __name__ == "__main__":
    main()
