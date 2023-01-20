def get_middle_point(x_pos, y_pos, w_len, h_len):
    return int(x_pos + w_len / 2), int(y_pos + h_len / 2)


def merge_boxes(boxes, offset_x=0, offset_y=0):
    results = []
    for box in enumerate(boxes):
        x_box, y_box, w_box, h_box = box[1]
        xm, ym = get_middle_point(x_box, y_box, w_box, h_box)
        inside = False
        for result in enumerate(results):
            x_res, y_res, w_res, h_res = result[1]
            xmr, ymr = get_middle_point(x_res, y_res, w_res, h_res)
            if xmr - offset_x < xm < xmr + offset_x and ymr - offset_y < ym < ymr + offset_y:
                inside = True
        if not inside:
            results.append(box[1])

    return results


class BoundingBox:
    pass
