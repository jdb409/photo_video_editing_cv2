import cv2


def outline(image, rect, color):
    if rect is None:
        return

    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x+w, y+h), color)


def copy_rect(src, dst, src_rect, dst_rect, interpolation=cv2.INTER_LINEAR):
    """
    copy part of the src to part of the dst.
    rects have to be of same size, so must resize first
    """

    x0, y0, w0, h0 = src_rect
    x1, y1, w1, h1 = dst_rect

    # resize contents of src sub rect
    # put result in dst sub rect
    src_resized = cv2.resize(
        src[y0:y0+h0, x0:x0+w0], (w1, h1), interpolation=interpolation)

    dst[y1:y1+h1, x1:x1+w1] = src_resized


def swap_rects(src, dst, rects,
               interpolation=cv2.INTER_LINEAR):
    """Copy the source with two or more sub-rectangles swapped.
        May not work for overlapping rectangles
    """

    if (dst is not src):
        dst[:] = src

    num_rects = len(rects)

    if (num_rects < 2):
        return

    x, y, w, h = rects[num_rects - 1]
    temp = src[y:y+h, x:x+w].copy()

    # copy contents of each rect into the next
    i = num_rects - 2
    while i >= 0:
        copy_rect(src, dst, rects[i], rects[i+1], interpolation)
        i -= 1

        # Copy the temporarily stored content into the first rectangle.
        copy_rect(temp, dst, (0, 0, w, h), rects[0], interpolation)
