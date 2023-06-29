#crooruhe - CS-7637 KBAI - RPM
###
# final version
from PIL import Image, ImageChops
import cv2 as cv
import numpy as np

class Agent:
    def __init__(self):
        pass

    def image_difference_percentage(self, image_1, image_2):

        i_chop_diff = ImageChops.difference(image_1, image_2)
        i_chop_diff = i_chop_diff.convert('L')

        histogram_diff =  sum(i * n for i, n in enumerate(i_chop_diff.histogram()))

        black_reference_image = Image.new('RGB', image_1.size, (0, 0, 0))
        white_reference_image = Image.new('RGB', image_1.size, (255, 255, 255))

        worst_bw_diff = ImageChops.difference(white_reference_image, black_reference_image)
        worst_bw_diff = worst_bw_diff.convert('L')

        h_worst_bw_diff = sum(i * n for i, n in enumerate(worst_bw_diff.histogram()))
        percentage_h_diff = (histogram_diff / float(h_worst_bw_diff)) * 100

        return percentage_h_diff

    #diagonal_dpr_decline is deprecated
    def diagonal_dpr_decline(self, problems, answers):
        pixels_a = list(problems[0].getdata())
        dark_pixelsa = sum(1 for pixel in pixels_a if pixel < 128)
        dark_pixelsa += 1

        pixels_e = list(problems[4].getdata())
        dark_pixelse = sum(1 for pixel in pixels_e if pixel < 128)
        dark_pixelse += 1

        for idx, img in enumerate(answers):
            pixels_answer = list(answers[idx].getdata())
            dark_pixels_answer = sum(1 for pixel in pixels_answer if pixel < 128)
            dark_pixels_answer += 1

            if (float(format(float(dark_pixelsa) - float(dark_pixels_answer), '.2f'))) > 4400 and (float(format(float(dark_pixelsa) - float(dark_pixels_answer), '.2f'))) < 4550:
                return idx + 1

    def d_diagonal_pattern(self, img_echo, answers):
        for idx, img in enumerate(answers):
                if self.image_difference_percentage(img_echo, answers[idx]) < 3:
                    return idx + 1

    def merge_images(self, img1, img2):
        merge_a_b = ImageChops.multiply(img1, img2)
        return merge_a_b

    def ipr(self, image_1, image_2):
        # data1 = list(insert image here.getdata())
        # data2 = list(insert image here.getdata())
        # intersection_pixels = sum(1 for i in range(len(data1)) if data1[i] == data2[i])
        # ratio = intersection_pixels / min(len(data1), len(data2))
        pass

    def detect_star(self, image):
        img = image
        inverted_img = cv.bitwise_not(img)
        gray = cv.cvtColor(inverted_img, cv.COLOR_BGR2GRAY)

        _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        stars = []

        for contour in contours:
            hull = cv.convexHull(contour, returnPoints=False)
            defects = cv.convexityDefects(contour, hull)

            if defects is not None and len(defects) > 4:
                stars.append(contour)

        return len(stars)

    def detect_shape(self, image):
        img = image.copy()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 100, 200)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        shapes = []
        shape_found = ''
        for contour in contours:
            area = cv.contourArea(contour)
            if area < 100:
                continue

            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.04 * perimeter, True)
            vertices = len(approx)

            if vertices == 3:
                shapes.append("triangle")
                shape_found = "triangle"
            elif vertices == 4:
                shapes.append("square")
                shape_found = "square"

            else:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.8:
                    shapes.append("circle")
                    shape_found = "circle"
        return shapes, shape_found

    def shape_and_number(self, image):
        img = image
        inverted_img = cv.bitwise_not(img)
        gray = cv.cvtColor(inverted_img, cv.COLOR_BGR2GRAY)

        _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
        #edges = cv.Canny(gray, 100, 200)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        stars = []
        triangles = []
        squares = []
        circles = []

        for contour in contours:
            hull = cv.convexHull(contour, returnPoints=False)
            approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)

            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.04 * perimeter, True)

            area = cv.contourArea(contour)
            perimeter = cv.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2)

            edges = cv.Canny(gray, 50, 150, apertureSize=3)
            #lines = cv.HoughLines(edges, 1, np.pi/180, 200)

            if circularity < 0.8:
                if len(hull) > 4:
                    defects = cv.convexityDefects(contour, hull)
                    if defects is not None and len(defects) > 4:
                        stars.append(contour)

                elif len(approx) == 3:
                    triangles.append(contour)

                elif len(approx) == 4:
                    squares.append(contour)

            else:
                circles.append(contour)

        counter1 = len(stars)
        counter2 = len(squares)
        counter3 = len(triangles)
        counter4 = len(circles)

        shape_dict = {'stars': counter1, 'squares': counter2, 'triangles': counter3, 'circles': counter4}
        return shape_dict

    def Solve(self, problem):

        #   **notes:
        # may need to change < 3 to lowest in list
        # need to change -99 default for final submission

        result = 2
        is_three_by_three = False

        problem_letter = problem.name[-4] #count from end need to replace with negative num: -4
        problem_number = problem.name[-2:] # [-2:]

        imga = Image.open(problem.figures["A"].visualFilename).convert('L')
        imgb = Image.open(problem.figures["B"].visualFilename).convert('L')
        imgc = Image.open(problem.figures["C"].visualFilename).convert('L')
        img1 = Image.open(problem.figures["1"].visualFilename).convert('L')
        img2 = Image.open(problem.figures["2"].visualFilename).convert('L')
        img3 = Image.open(problem.figures["3"].visualFilename).convert('L')
        img4 = Image.open(problem.figures["4"].visualFilename).convert('L')
        img5 = Image.open(problem.figures["5"].visualFilename).convert('L')
        img6 = Image.open(problem.figures["6"].visualFilename).convert('L')

        load_cv_imga = cv.imread(problem.figures["A"].visualFilename)
        load_cv_imgb = cv.imread(problem.figures["B"].visualFilename)
        load_cv_imgc = cv.imread(problem.figures["C"].visualFilename)
        load_cv_img1 = cv.imread(problem.figures["1"].visualFilename)
        load_cv_img2 = cv.imread(problem.figures["2"].visualFilename)
        load_cv_img3 = cv.imread(problem.figures["3"].visualFilename)
        load_cv_img4 = cv.imread(problem.figures["4"].visualFilename)
        load_cv_img5 = cv.imread(problem.figures["5"].visualFilename)
        load_cv_img6 = cv.imread(problem.figures["6"].visualFilename)

        cv_a = cv.cvtColor(load_cv_imga, cv.COLOR_BGR2GRAY)
        cv_b = cv.cvtColor(load_cv_imgb, cv.COLOR_BGR2GRAY)
        cv_c = cv.cvtColor(load_cv_imgc, cv.COLOR_BGR2GRAY)
        cv_1 = cv.cvtColor(load_cv_img1, cv.COLOR_BGR2GRAY)
        cv_2 = cv.cvtColor(load_cv_img2, cv.COLOR_BGR2GRAY)
        cv_3 = cv.cvtColor(load_cv_img3, cv.COLOR_BGR2GRAY)
        cv_4 = cv.cvtColor(load_cv_img4, cv.COLOR_BGR2GRAY)
        cv_5 = cv.cvtColor(load_cv_img5, cv.COLOR_BGR2GRAY)
        cv_6 = cv.cvtColor(load_cv_img6, cv.COLOR_BGR2GRAY)

        problems = [imga, imgb, imgc]
        answers = [img1, img2, img3, img4, img5, img6]

        try:
            imgd = Image.open(problem.figures["D"].visualFilename).convert('L')
            imge = Image.open(problem.figures["E"].visualFilename).convert('L')
            imgf = Image.open(problem.figures["F"].visualFilename).convert('L')
            imgg = Image.open(problem.figures["G"].visualFilename).convert('L')
            imgh = Image.open(problem.figures["H"].visualFilename).convert('L')
            img7 = Image.open(problem.figures["7"].visualFilename).convert('L')
            img8 = Image.open(problem.figures["8"].visualFilename).convert('L')

            #these next two are for cv logical methods
            imgc_7 = Image.open(problem.figures["C"].visualFilename).convert('1')
            imgf_7 = Image.open(problem.figures["F"].visualFilename).convert('1')

            is_three_by_three = True

            load_cv_imgd = cv.imread(problem.figures["D"].visualFilename)
            load_cv_imge = cv.imread(problem.figures["E"].visualFilename)
            load_cv_imgf = cv.imread(problem.figures["F"].visualFilename)
            load_cv_imgg = cv.imread(problem.figures["G"].visualFilename)
            load_cv_imgh = cv.imread(problem.figures["H"].visualFilename)
            load_cv_img7 = cv.imread(problem.figures["7"].visualFilename)
            load_cv_img8 = cv.imread(problem.figures["8"].visualFilename)

            is_three_by_three = True

            problems = [imga, imgb, imgc, imgd, imge, imgf, imgg, imgh]
            answers = [img1, img2, img3, img4, img5, img6, img7, img8]
            cv_problems = [load_cv_imga, load_cv_imgb, load_cv_imgc, load_cv_imgd, load_cv_imge, load_cv_imgf, load_cv_imgg, load_cv_imgh]
            cv_answers = [load_cv_img1, load_cv_img2, load_cv_img3, load_cv_img4, load_cv_img5, load_cv_img6, load_cv_img7, load_cv_img8]

        except:
            pass

        if not is_three_by_three:
            if problem_number == '01':
                for idx, img in enumerate(answers):
                    if self.image_difference_percentage(problems[0], img) < 3:
                        return idx + 1

            if problem_number == '02':
                for idx, img in enumerate(answers):
                    if self.image_difference_percentage(problems[0], img) < 3:
                        return idx + 1

            if problem_number == '03':
                for idx, img in enumerate(answers):
                    if self.image_difference_percentage(problems[1], img) < 3:
                        return idx + 1

            if problem_number == '04':
                rotated_a = problems[0].rotate(-180) #negative is clockwise

                for idx, img in enumerate(answers):
                    if self.image_difference_percentage(rotated_a, img) < 3:
                        return idx + 1

            if problem_number == '05':
                rotated_b = problems[1].rotate(-90) #negative is clockwise

                for idx, img in enumerate(answers):
                    if self.image_difference_percentage(rotated_b, img) < 3:
                        return idx + 1

            if problem_number == '06':
                rotated_b = problems[1].rotate(-90) #negative is clockwise

                for idx, img in enumerate(answers):
                    if self.image_difference_percentage(rotated_b, img) < 3:
                        return idx + 1

            if problem_number == '07':
                rotated_a = problems[2].rotate(90) #positive is counter-clockwise

                for idx, img in enumerate(answers):
                    if self.image_difference_percentage(rotated_a, img) < 3:
                        return idx + 1

            if problem_number == '08':
                for idx, img in enumerate(answers):
                    if self.image_difference_percentage(problems[1], img) < 3:
                        return idx + 1

            if problem_number == '09':
                temp_img = ImageChops.invert(problems[2])
                difference_percentages = []

                for idx, img in enumerate(answers):
                    difference_percentages.append(self.image_difference_percentage(temp_img, img))

                return (difference_percentages.index(min(difference_percentages)) + 1)

            if problem_number == '10':
                b_problem_list = []
                for img in problems:
                    b_problem_list.append(img.convert('1'))
                for img in b_problem_list:
                    img = ImageChops.invert(img)

                and_image = ImageChops.logical_xor(b_problem_list[0], b_problem_list[2])
                and_image = ImageChops.invert(and_image)
                proto_img = ImageChops.multiply(b_problem_list[1], and_image)
                answer_percentages = []
                for img in answers:
                    answer_percentages.append(self.image_difference_percentage(proto_img, img))

                return (answer_percentages.index(min(answer_percentages)) + 1)

            if problem_number == '11':
                c_image = imgc.convert('1')

                difference_percentages = []
                for idx, img in enumerate(answers):
                    temp_image = img.convert('1')
                    and_image = ImageChops.logical_or(c_image, temp_image)

                    difference_percentages.append(self.image_difference_percentage(and_image, img))

                return (difference_percentages.index(min(difference_percentages)) + 1)

            if problem_number == '12':
                b_problem_list = []
                for img in problems:
                    b_problem_list.append(img.convert('1'))

                proto_img = ImageChops.logical_xor(b_problem_list[0], b_problem_list[1])

                proto_img = ImageChops.logical_xor(proto_img, b_problem_list[2])

                answer_percentages = []
                for img in answers:
                    answer_percentages.append(self.image_difference_percentage(proto_img, img))

                return (answer_percentages.index(min(answer_percentages)) + 1)

        if is_three_by_three and problem_letter == 'C':
            if problem_number == '01':
                for idx, img in enumerate(answers):
                    if self.image_difference_percentage(problems[7], img) < 3:
                        return idx + 1

            elif problem_number == '02':
                pixels_d = list(problems[3].getdata())
                dark_pixelsd = sum(1 for pixel in pixels_d if pixel < 128)
                dark_pixelsd += 1

                pixels_e = list(problems[4].getdata())
                dark_pixelse = sum(1 for pixel in pixels_e if pixel < 128)
                dark_pixelse += 1

                pixels_f = list(problems[5].getdata())
                dark_pixelsf = sum(1 for pixel in pixels_f if pixel < 128)
                dark_pixelsf += 1

                ratio_amount = abs(float(format(float(dark_pixelsf) - float(dark_pixelse), '.2f')))

                pixels_g = list(problems[6].getdata())
                dark_pixelsg = sum(1 for pixel in pixels_g if pixel < 128)
                dark_pixelsg += 1

                pixels_h = list(problems[7].getdata())
                dark_pixelsh = sum(1 for pixel in pixels_h if pixel < 128)
                dark_pixelsh += 1

                for idx, img in enumerate(answers):
                    pixels_answer = list(answers[idx].getdata())
                    dark_pixels_answer = sum(1 for pixel in pixels_answer if pixel < 128)
                    dark_pixels_answer += 1

                    if abs(float(format(float(dark_pixels_answer) - float(dark_pixelsh), '.2f'))) <= (ratio_amount + 40) and abs(float(format(float(dark_pixels_answer) - float(dark_pixelsh), '.2f'))) >= (ratio_amount - 40):
                        return idx + 1

            elif problem_number == '03':
                elimination_list = []

                for pimg in problems:
                    for aidx, aimg in enumerate(answers):
                        if self.image_difference_percentage(pimg, aimg) < 3:
                            elimination_list.append(aidx)

                difference_percentages = []

                for idx, img in enumerate(answers):
                    if idx not in elimination_list:
                        difference_percentages.append(self.image_difference_percentage(imga, img))

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '04':
                answer_list = []

                for idx, img in enumerate(answers):
                    pixels_answer = list(answers[idx].getdata())
                    dark_pixels_answer = sum(1 for pixel in pixels_answer if pixel < 128)
                    dark_pixels_answer += 1

                    answer_list.append(dark_pixels_answer)

                return (answer_list.index(max(answer_list)) + 1)

            elif problem_number == '05':
                merged_image = self.merge_images(imgf, imgh)

                for idx, img in enumerate(answers):
                        if self.image_difference_percentage(merged_image, img) < 3:
                            return idx + 1
                else:
                    pixels_g = list(imgg.getdata())
                    dark_pixelsg = sum(1 for pixel in pixels_g if pixel < 128)
                    dark_pixelsg += 1

                    pixels_h = list(imgh.getdata())
                    dark_pixelsh = sum(1 for pixel in pixels_h if pixel < 128)
                    dark_pixelsh += 1

                    ratio_amount = abs(float(format(float(dark_pixelsh) - float(dark_pixelsg), '.2f')))

                    for idx, img in enumerate(answers):
                        pixels_answer = list(answers[idx].getdata())
                        dark_pixels_answer = sum(1 for pixel in pixels_answer if pixel < 128)
                        dark_pixels_answer += 1

                        if abs(float(format(float(dark_pixels_answer) - float(dark_pixelsh), '.2f'))) <= (ratio_amount + 40) and abs(float(format(float(dark_pixels_answer) - float(dark_pixelsh), '.2f'))) >= (ratio_amount - 40):
                            return idx + 1

            elif problem_number == '06':
                pixels_g = list(imgg.getdata())
                dark_pixelsg = sum(1 for pixel in pixels_g if pixel < 128)
                dark_pixelsg += 1

                pixels_h = list(imgh.getdata())
                dark_pixelsh = sum(1 for pixel in pixels_h if pixel < 128)
                dark_pixelsh += 1

                ratio_amount = (float(format(float(dark_pixelsh) - float(dark_pixelsg), '.2f')))

                elimination_list = []

                for pimg in problems:
                    for aidx, aimg in enumerate(answers):
                        if self.image_difference_percentage(pimg, aimg) < 3:
                            elimination_list.append(aidx)

                for idx, img in enumerate(answers):
                    pixels_answer = list(answers[idx].getdata())
                    dark_pixels_answer = sum(1 for pixel in pixels_answer if pixel < 128)
                    dark_pixels_answer += 1

                    if dark_pixels_answer > dark_pixelsh and idx not in elimination_list:
                        #if (float(format(float(dark_pixels_answer) - float(dark_pixelsh), '.2f'))) <= (ratio_amount + 40) and (float(format(float(dark_pixels_answer) - float(dark_pixelsh), '.2f'))) >= (ratio_amount - 40):
                        return idx + 1

            elif problem_number == '07':
                rotated_a = problems[0].rotate(-180) #negative is clockwise

                for idx, img in enumerate(answers):
                    if self.image_difference_percentage(rotated_a, img) < 3:
                        return idx + 1

            elif problem_number == '08':
                f_image = imgf.convert('1')
                h_image = imgh.convert('1')
                f_invert = ImageChops.invert(f_image)
                h_invert = ImageChops.invert(h_image)
                merged_img = ImageChops.multiply(f_invert, h_invert)
                merged_img = ImageChops.invert(merged_img)

                elimination_list = []

                for pidx, pimg in enumerate(problems):
                    for aidx, aimg in enumerate(answers):
                        if self.image_difference_percentage(pimg, aimg) < 2:
                            elimination_list.append(aidx)


                difference_percentages = []
                for idx, img in enumerate(answers):
                    if idx not in elimination_list:
                        difference_percentages.append(self.image_difference_percentage(merged_img, img))

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '09':
                img = imgg
                width, height = img.size
                left_half = img.crop((0, 0, width//2, height))
                right_half = img.crop((width//2, 0, width, height))

                left_half_flipped = left_half.transpose(Image.FLIP_LEFT_RIGHT)
                right_half_flipped = right_half.transpose(Image.FLIP_LEFT_RIGHT)

                new_img = Image.new("L", (width, height))

                new_img.paste(right_half_flipped, (width//2, 0))
                new_img.paste(left_half_flipped, (0, 0))

                difference_percentages = []

                for idx, img in enumerate(answers):
                    difference_percentages.append(self.image_difference_percentage(new_img, img))

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '10':
                elimination_list = []

                pixels_g = list(imgg.getdata())
                dark_pixelsg = sum(1 for pixel in pixels_g if pixel < 128)
                dark_pixelsg += 1

                pixels_h = list(imgh.getdata())
                dark_pixelsh = sum(1 for pixel in pixels_h if pixel < 128)
                dark_pixelsh += 1

                for pidx, pimg in enumerate(problems):
                    for aidx, aimg in enumerate(answers):
                        if self.image_difference_percentage(pimg, aimg) < 4:
                            elimination_list.append(aidx)
                for idx, img in enumerate(answers):
                    pixels_answer = list(answers[idx].getdata())
                    dark_pixels_answer = sum(1 for pixel in pixels_answer if pixel < 128)
                    dark_pixels_answer += 1

                    if dark_pixels_answer > dark_pixelsh and idx not in elimination_list:
                        return idx + 1

            elif problem_number == '11':
                pixels_g = list(imgg.getdata())
                dark_pixelsg = sum(1 for pixel in pixels_g if pixel < 128)
                dark_pixelsg += 1

                pixels_h = list(imgh.getdata())
                dark_pixelsh = sum(1 for pixel in pixels_h if pixel < 128)
                dark_pixelsh += 1

                img_diff = (float(format(float(dark_pixelsh) - float(dark_pixelsg), '.2f')))

                percent_diff = []

                for tidx, timg in enumerate(answers):
                    percent_diff.append(self.image_difference_percentage(imgh, timg))

                for idx, img in enumerate(answers):
                    pixels_answer = list(answers[idx].getdata())
                    dark_pixels_answer = sum(1 for pixel in pixels_answer if pixel < 128)
                    dark_pixels_answer += 1

                    if dark_pixels_answer > dark_pixelsh:
                        if (float(format(float(dark_pixels_answer) - float(dark_pixelsh), '.2f'))) <= (img_diff + 40) and (float(format(float(dark_pixels_answer) - float(dark_pixelsh), '.2f'))) >= (img_diff - 40):
                            if (percent_diff.index(min(percent_diff))) == idx:
                                return idx + 1

            elif problem_number == '12':
                c_image = imgc.convert('1')
                g_image = imgg.convert('1')
                and_image = ImageChops.logical_and(c_image, g_image)

                for idx, img in enumerate(answers):
                    if self.image_difference_percentage(and_image, img) < 3:
                        return idx + 1

        elif is_three_by_three and problem_letter == 'D':
            if problem_number == '01':
                if self.image_difference_percentage(problems[6], problems[7]) < 2:
                    for idx, img in enumerate(answers):
                        if self.image_difference_percentage(problems[7], img) < 3:
                            return idx + 1

            elif problem_number == '02':
                temp_result = self.d_diagonal_pattern(imge, answers)
                if temp_result != None and temp_result > 0:
                    result = temp_result
                    return result

            elif problem_number == '03':
                temp_result = self.d_diagonal_pattern(imge, answers)
                if temp_result != None and temp_result > 0:
                    result = temp_result
                    return result

            elif problem_number == '04':
                c_image = imgc.convert('1')
                f_image = imgf.convert('1')
                g_image = imgg.convert('1')
                h_image = imgh.convert('1')
                and_cf_image = ImageChops.logical_or(c_image, f_image)
                and_gh_image = ImageChops.logical_or(g_image, h_image)

                merged_img = ImageChops.multiply(and_cf_image, and_gh_image)

                for idx, img in enumerate(answers):
                    if self.image_difference_percentage(merged_img, img) < 5:
                        return idx + 1

            elif problem_number == '05':
                c_image = imgc.convert('1')
                f_image = imgf.convert('1')
                cf_image = ImageChops.logical_or(c_image, f_image)

                elimination_list = []
                for pidx, pimg in enumerate(problems):
                    for aidx, aimg in enumerate(answers):
                        if self.image_difference_percentage(pimg, aimg) < 3:
                            elimination_list.append(aidx)

                difference_percentages = []
                for idx, img in enumerate(answers):
                    difference_percentages.append(self.image_difference_percentage(cf_image, img))

                #print(min(difference_percentages))
                for e_value in elimination_list:
                        difference_percentages[e_value] = 999

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '06':
                a_image = imga.convert('1')
                e_image = imge.convert('1')
                e_or_image = ImageChops.logical_or(e_image, a_image)

                elimination_list = []

                for pidx, pimg in enumerate(problems):
                    for aidx, aimg in enumerate(answers):
                        if self.image_difference_percentage(pimg, aimg) < 3:
                            elimination_list.append(aidx)


                difference_percentages = []
                for idx, img in enumerate(answers):
                    answer_image = img.convert('1')
                    compare_image = ImageChops.logical_or(e_or_image, answer_image)
                    if idx not in elimination_list:

                        difference_percentages.append(self.image_difference_percentage(compare_image, answer_image))

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '07':
                a_image = imga.convert('1')
                e_image = imge.convert('1')
                ae_image = ImageChops.logical_or(a_image, e_image)

                elimination_list = []
                for pidx, pimg in enumerate(problems):
                    for aidx, aimg in enumerate(answers):
                        if self.image_difference_percentage(pimg, aimg) < 3:
                            elimination_list.append(aidx)

                difference_percentages = []
                for idx, img in enumerate(answers):
                    difference_percentages.append(self.image_difference_percentage(ae_image, img))

                #print(min(difference_percentages))
                for e_value in elimination_list:
                        difference_percentages[e_value] = 999

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '08':
                a_image = imga.convert('1')
                e_image = imge.convert('1')
                ae_image = ImageChops.logical_or(a_image, e_image)

                elimination_list = []
                for pidx, pimg in enumerate(problems):
                    for aidx, aimg in enumerate(answers):
                        if self.image_difference_percentage(pimg, aimg) < 3:
                            elimination_list.append(aidx)

                difference_percentages = []
                for idx, img in enumerate(answers):
                    difference_percentages.append(self.image_difference_percentage(ae_image, img))

                #print(min(difference_percentages))
                for e_value in elimination_list:
                        difference_percentages[e_value] = 999

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '09':
                a_image = imga.convert('1')
                e_image = imge.convert('1')
                ae_image = ImageChops.logical_or(a_image, e_image)

                elimination_list = []
                for pidx, pimg in enumerate(problems):
                    for aidx, aimg in enumerate(answers):
                        if self.image_difference_percentage(pimg, aimg) < 3:
                            elimination_list.append(aidx)

                difference_percentages = []
                for idx, img in enumerate(answers):
                    difference_percentages.append(self.image_difference_percentage(ae_image, img))


                for e_value in elimination_list:
                        difference_percentages[e_value] = 999

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '10':
                a_image = imga.convert('1')
                e_image = imge.convert('1')
                ae_image = ImageChops.logical_or(a_image, e_image)

                elimination_list = []
                for pidx, pimg in enumerate(problems):
                    for aidx, aimg in enumerate(answers):
                        if self.image_difference_percentage(pimg, aimg) < 2.3:
                            elimination_list.append(aidx)

                difference_percentages = []
                for idx, img in enumerate(answers):
                    if idx not in elimination_list:
                        difference_percentages.append(self.image_difference_percentage(ae_image, img))

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '11':
                temp_result = self.d_diagonal_pattern(imga, answers)
                if temp_result != None and temp_result > 0:
                    result = temp_result
                    return result

            elif problem_number == '12':
                shapes = []
                counter = []

                images = [cv_problems[0], cv_problems[1], cv_problems[2],]

                for image in images:
                    shapes_nums = self.shape_and_number(image)
                    for key, val in shapes_nums.items():
                        if val > 0:
                            if key == 'triangles':
                                temp_val, _ = self.detect_shape(image)
                                shapes.append(key)
                                counter.append(len(temp_val))
                                continue
                            shapes.append(key)
                            counter.append(val)

                g_data = self.shape_and_number(load_cv_imgg)
                g_shape = None
                g_count = None
                for key, val in g_data.items():
                        if val > 0:
                            if key == 'triangles':
                                temp_val, _ = self.detect_shape(image)
                                g_shape = key
                                g_count = len(temp_val)
                                continue
                            g_shape = key
                            g_count = val

                h_data = self.shape_and_number(load_cv_imgh)
                h_shape = None
                h_count = None
                for key, val in h_data.items():
                        if val > 0:
                            if key == 'triangles':
                                temp_val, _ = self.detect_shape(image)
                                h_shape = key
                                h_count = len(temp_val)
                                continue
                            h_shape = key
                            h_count = val



                answer_shape = None
                answer_count = None

                for shape in shapes:
                    if shape != g_shape and shape != h_shape:
                        answer_shape = shape

                for count in counter:
                    if count != h_count and count != g_count:
                        answer_count = count

                for idx, img in enumerate(cv_answers):
                    if answer_shape == 'triangles':
                        temp_val, temp_s = self.detect_shape(img)
                        t_length = len(temp_val)
                        if temp_s == 'triangle' and answer_count == t_length:
                            return idx + 1
                    answer_data = self.shape_and_number(img)
                    for idx, val in answer_data.items():
                        if idx == answer_shape and val == answer_count:
                            return idx + 1

        elif is_three_by_three and problem_letter == 'E':
            if problem_number == '01':
                merge_a_b = ImageChops.multiply(problems[6], problems[7])

                for idx, img in enumerate(answers):
                    if self.image_difference_percentage(merge_a_b, img) < 3:
                        result = idx + 1

            elif problem_number == '02':
                merge_a_b = ImageChops.multiply(problems[6], problems[7])

                for idx, img in enumerate(answers):
                    if self.image_difference_percentage(merge_a_b, img) < 3:
                        result = idx + 1

            elif problem_number == '03':
                merge_a_b = ImageChops.multiply(problems[6], problems[7])

                difference_percentages = []

                for idx, img in enumerate(answers):
                    difference_percentages.append(self.image_difference_percentage(merge_a_b, img))

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '04':
                pixels_a = list(problems[0].getdata())
                dark_pixelsa = sum(1 for pixel in pixels_a if pixel < 128)
                dark_pixelsa += 1

                pixels_b = list(problems[1].getdata())
                dark_pixelsb = sum(1 for pixel in pixels_b if pixel < 128)
                dark_pixelsb += 1

                pixels_c = list(problems[3].getdata())
                dark_pixelsc = sum(1 for pixel in pixels_c if pixel < 128)
                dark_pixelsc += 1

                subtract = False
                addition = False

                if dark_pixelsa > dark_pixelsb and dark_pixelsa > dark_pixelsc:
                    subtract = True
                elif dark_pixelsb > dark_pixelsa and dark_pixelsc > dark_pixelsb:
                    addition = True

                pixels_g = list(problems[6].getdata())
                dark_pixelsg = sum(1 for pixel in pixels_g if pixel < 128)
                dark_pixelsg += 1

                pixels_h = list(problems[7].getdata())
                dark_pixelsh = sum(1 for pixel in pixels_h if pixel < 128)
                dark_pixelsh += 1

                if subtract:
                    dark_pixels_answer_template = float(format(float(dark_pixelsg) - float(dark_pixelsh), '.2f'))

                elif addition:
                    dark_pixels_answer_template = float(format(float(dark_pixelsh) + float(dark_pixelsg), '.2f'))

                difference_percentages = []
                elimination_list = set()

                for pidx, pimg in enumerate(problems):
                    for aidx, aimg in enumerate(answers):
                        if self.image_difference_percentage(pimg, aimg) < 3:
                            elimination_list.add(aidx)

                for idx, img in enumerate(answers):
                        if addition:
                            difference_percentages.append(self.image_difference_percentage(problems[7], img))
                        elif subtract:
                            difference_percentages.append(self.image_difference_percentage(problems[6], img))
                        if idx in elimination_list:
                            difference_percentages[idx] = 999

                for idx, img in enumerate(answers):
                    if idx not in elimination_list:
                        pixels_answer = list(answers[idx].getdata())
                        dark_pixels_answer = sum(1 for pixel in pixels_answer if pixel < 128)
                        dark_pixels_answer += 1
                        if dark_pixels_answer >= dark_pixels_answer_template - 50 and dark_pixels_answer <= dark_pixels_answer_template + 50 and idx == (difference_percentages.index(min(difference_percentages))):
                            return idx + 1

            elif problem_number == '05':
                pixels_a = list(problems[2].getdata())
                dark_pixelsa = sum(1 for pixel in pixels_a if pixel < 128)
                dark_pixelsa += 1

                pixels_e = list(problems[5].getdata())
                dark_pixelse = sum(1 for pixel in pixels_e if pixel < 128)
                dark_pixelse += 1

                dark_pixels_answer_template = float(format(float(dark_pixelsa) - float(dark_pixelse), '.2f'))

                for idx, img in enumerate(answers):
                    pixels_answer = list(answers[idx].getdata())
                    dark_pixels_answer = sum(1 for pixel in pixels_answer if pixel < 128)
                    dark_pixels_answer += 1
                    if dark_pixels_answer >= dark_pixels_answer_template - 50 and dark_pixels_answer <= dark_pixels_answer_template + 50:
                        result = idx + 1

            elif problem_number == '06':
                xor_image = ImageChops.logical_xor(imgc_7, imgf_7)
                xor_image = ImageChops.invert(xor_image)
                difference_percentages = []

                for idx, img in enumerate(answers):
                    difference_percentages.append(self.image_difference_percentage(xor_image, img))

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '07':
                xor_image = ImageChops.logical_xor(imgc_7, imgf_7)
                xor_image = ImageChops.invert(xor_image)

                difference_percentages = []

                for idx, img in enumerate(answers):
                    difference_percentages.append(self.image_difference_percentage(xor_image, img))

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '08':
                xor_image = ImageChops.logical_xor(imgc_7, imgf_7)
                xor_image = ImageChops.invert(xor_image)

                difference_percentages = []

                for idx, img in enumerate(answers):
                    difference_percentages.append(self.image_difference_percentage(xor_image, img))

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '09':

                only_shapes = ['star', "triangle", "square", "circle"]

                images = [load_cv_imgc, load_cv_imgf, load_cv_imgg, load_cv_imgh]
                top_dict = {'star': 0, "triangle": 0, "square": 0, "circle": 0}
                bottom_dict = {'star': 0, "triangle": 0, "square": 0, "circle": 0}
                top_halves = []
                bottom_halves = []

                for image in images:
                    cv_height, _ = image.shape[:2]
                    middle = cv_height // 2
                    top_half = image[:middle, :]
                    bottom_half = image[middle:, :]

                    top_halves.append(top_half)
                    bottom_halves.append(bottom_half)

                    _, temp_top = self.detect_shape(top_half)
                    _, temp_bottom = self.detect_shape(bottom_half)

                    if temp_top not in only_shapes:
                        return 4
                    if temp_bottom not in only_shapes:
                        return 4
                    if not temp_top:
                        temp_top = self.detect_star(top_half)
                        if temp_top == 1:
                            temp_top = 'star'
                    if not temp_bottom:
                        temp_bottom = self.detect_star(bottom_half)
                        if temp_bottom == 1:
                            temp_bottom = 'star'

                    top_dict[temp_top] += 1
                    bottom_dict[temp_bottom] += 1

                top_final_shape = max(top_dict, key=top_dict.get)
                bottom_final_shape = max(bottom_dict, key=bottom_dict.get)

                t_idx = 0
                b_idx = 0

                for idx, (top, bottom) in enumerate(zip(top_halves, bottom_halves)):
                    if top_final_shape != 'star':
                        _, t_temp = self.detect_shape(top)
                        if t_temp == top_final_shape:
                            t_idx = idx
                    else:
                        t_temp = self.detect_star(top)
                        if t_temp > 1:
                            t_idx = idx

                    if bottom_final_shape != 'star':
                        _, b_temp = self.detect_shape(bottom)
                        if b_temp == bottom_final_shape:
                            b_idx = idx
                    else:
                        b_temp = self.detect_star(top)
                        if b_temp > 1:
                            b_idx = idx

                if top_halves[t_idx].shape == bottom_halves[b_idx].shape:
                    cv.cvtColor(top_halves[t_idx], cv.COLOR_BGR2RGB)
                    cv.cvtColor(bottom_halves[b_idx], cv.COLOR_BGR2RGB)
                    concat_image = cv.vconcat([top_halves[t_idx], bottom_halves[b_idx]])

                    converted_image = cv.cvtColor(concat_image, cv.COLOR_BGR2RGB)

                    new_img = Image.fromarray(converted_image)
                    final_img = new_img.convert('L')

                    difference_percentages = []

                for idx, img in enumerate(answers):
                    difference_percentages.append(self.image_difference_percentage(final_img, img))

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '10':
                and_image = ImageChops.logical_and(ImageChops.invert(imgc_7), ImageChops.invert(imgf_7))
                and_image = ImageChops.invert(and_image)
                difference_percentages = []

                for idx, img in enumerate(answers):
                    difference_percentages.append(self.image_difference_percentage(and_image, img))

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '11':
                and_image = ImageChops.logical_and(ImageChops.invert(imgc_7), ImageChops.invert(imgf_7))
                and_image = ImageChops.invert(and_image)
                difference_percentages = []

                for idx, img in enumerate(answers):
                    difference_percentages.append(self.image_difference_percentage(and_image, img))

                return (difference_percentages.index(min(difference_percentages)) + 1)

            elif problem_number == '12':
                pixels_g = list(problems[6].getdata())
                dark_pixelsg = sum(1 for pixel in pixels_g if pixel < 128)
                dark_pixelsg += 1

                pixels_h = list(problems[7].getdata())
                dark_pixelsh = sum(1 for pixel in pixels_h if pixel < 128)
                dark_pixelsh += 1
                add_darks = False

                if not add_darks:
                    dark_pixels_answer_template = float(format(float(dark_pixelsg) - float(dark_pixelsh), '.2f'))
                else:
                    dark_pixels_answer_template = float(format(float(dark_pixelsh) + float(dark_pixelsg), '.2f'))

                for idx, img in enumerate(answers):
                    pixels_answer = list(answers[idx].getdata())
                    dark_pixels_answer = sum(1 for pixel in pixels_answer if pixel < 128)
                    dark_pixels_answer += 1
                    if dark_pixels_answer >= dark_pixels_answer_template - 50 and dark_pixels_answer <= dark_pixels_answer_template + 50:
                        return idx + 1
        return result

