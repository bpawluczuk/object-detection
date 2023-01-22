def get_middle_point(x_pos, y_pos, w_len, h_len):
    return int(x_pos + w_len / 2), int(y_pos + h_len / 2)


def merge_boxes(
        boxes,
        offset_x=0,
        offset_y=0,
):
    results = []
    for i_box, box in enumerate(boxes):

        x_box, y_box, w_box, h_box = box
        xm, ym = get_middle_point(x_box, y_box, w_box, h_box)

        inside = False

        for i_res, result in enumerate(results):

            x_res, y_res, w_res, h_res = result
            xmr, ymr = get_middle_point(x_res, y_res, w_res, h_res)

            if xmr - offset_x < xm < xmr + offset_x and ymr - offset_y < ym < ymr + offset_y:
                inside = True

        if not inside:
            results.append(box)

    return results


def sort_boxes(rectangles, result=None):
    if result is None:
        result = []

    _, _, zw, zh = rectangles[0]
    area = zw * zh

    temp_box = ()

    for _, i_box in enumerate(rectangles):
        _, _, iw, ih = i_box
        i_area = iw * ih

        # sort asc, desc
        if area > i_area:
            area = i_area
            temp_box = i_box

    if temp_box:
        rectangles.remove(temp_box)
        result.append(temp_box)
        sort_boxes(rectangles, result)

    return result


class BoundingBox:
    pass
