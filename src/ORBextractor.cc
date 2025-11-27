/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/
/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

#include "ORBextractor.h"


using namespace cv;
using namespace std;

namespace ORB_SLAM3
{


const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;


/**
 * @brief Compute keypoint orientation using the intensity centroid (IC) method.
 *
 * The orientation is computed inside a circular patch of radius HALF_PATCH_SIZE
 * centered at @p pt.
 *
 * @param[in] image  Input grayscale image.
 * @param[in] pt     Keypoint position (in image coordinates).
 * @param[in] u_max  Precomputed horizontal radius for each row of the circular patch.
 * @return float     Orientation angle in degrees in [0, 360).
 */
static float IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
{

    int m_01 = 0, m_10 = 0;


    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));


    // The central line v = 0 must be handled separately.
    // Since we use the center row plus symmetric rows, PATCH_SIZE must be odd.
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        // Note that center[u] can be indexed with negative u because we work on
        // a patch centered at 'center'. Each pixel on the horizontal center
        // line is weighted by its x coordinate (u).
        m_10 += u * center[u];

    // Go line by line in the circular patch
    // 'step' is the number of bytes per image row.
    int step = (int)image.step1();
    // We iterate symmetrically over pairs of rows around v = 0
    // to reuse computations and speed up the process.
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        // m_01 is computed column-by-column, but due to symmetry and sign of
        // x,y, we can process the two symmetric rows at once.
        int v_sum = 0;
        // Retrieve max horizontal range for this row; the patch is circular.
        int d = u_max[v];
        // For each valid u in this row, we process two pixels at once:
        // one below the center line (x, y) and one above (x, -y).
        // m_10 = Σ x*I(x,y) = x*I(x,y) + x*I(x,-y) = x*(I(x,y) + I(x,-y))
        // m_01 = Σ y*I(x,y) = y*I(x,y) - y*I(x,-y) = y*(I(x,y) - I(x,-y))
        for (int u = -d; u <= d; ++u)
        {
            // val_plus  : intensity at (u, +v)
            // val_minus : intensity at (u, -v)
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            // Difference of intensities along the two symmetric rows (for m_01)
            v_sum += (val_plus - val_minus);
            // Weighted sum along x direction (for m_10) using u (can be negative)
            m_10 += u * (val_plus + val_minus);
        }
        // Accumulate for this pair of rows, weighted by y = v
        m_01 += v * v_sum;
    }

    // Use fastAtan2() (returns angle in degrees in [0,360), precision about 0.3°)
    return fastAtan2((float)m_01, (float)m_10);
}

/// Conversion factor from degrees to radians
const float factorPI = (float)(CV_PI/180.f);

/**
 * @brief Compute the ORB descriptor for one keypoint.
 *
 * This is a file-local static function and is only used inside this file.
 *
 * @param[in] kpt       Keypoint.
 * @param[in] img       Image from which the keypoint was extracted.
 * @param[in] pattern   Predefined random sampling pattern.
 * @param[out] desc     Output descriptor buffer (32 bytes = 256 bits).
 */
static void computeOrbDescriptor(const KeyPoint& kpt,
                                 const Mat& img, const Point* pattern,
                                 uchar* desc)
{
    // Get keypoint orientation (in radians). kpt.angle is in degrees [0,360).
    float angle = (float)kpt.angle*factorPI;
    // Precompute cos and sin for the rotation.
    float a = (float)cos(angle), b = (float)sin(angle);

    // Pointer to the center pixel of the patch.
    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    // Bytes per row.
    const int step = (int)img.step;

    // Original BRIEF is not rotation invariant. ORB uses a steered BRIEF, i.e.,
    // the sampling pattern is rotated by the keypoint orientation to obtain
    // rotation invariance.
    // For a pattern point (x,y), after rotation:
    //   x' = x*cos(θ) - y*sin(θ)
    //   y' = x*sin(θ) + y*cos(θ)
#define GET_VALUE(idx) center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + cvRound(pattern[idx].x*a - pattern[idx].y*b)]
    // y' * step
    // x'
    // ORB descriptor is 32 * 8 bits.
    // Each bit is the comparison of two pixel intensities. Each byte (8 bits)
    // therefore uses 16 sampling points, hence 'pattern += 16' per iteration.
    for (int i = 0; i < 32; ++i, pattern += 16)
    {

        int t0,     // intensity of first sample point
            t1,     // intensity of second sample point
            val;    // accumulated bits for this descriptor byte

        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;                          // bit 0
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;                  // bit 1
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;                  // bit 2
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;                  // bit 3
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;                  // bit 4
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;                  // bit 5
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;                  // bit 6
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;                  // bit 7

        // Store this descriptor byte.
        desc[i] = (uchar)val;
    } // Final ORB descriptor is 32 * 8 = 256 bits.

    // Undefine the macro to avoid conflicts elsewhere.
#undef GET_VALUE
}

// Predefined sampling pattern. 256 means 256 bits in the descriptor.
// Each bit uses one pair of points (2), and each point has 2 coordinates (x,y).
static int bit_pattern_31_[256*4] =
        {
                8,-3, 9,5/*mean (0), correlation (0)*/,
                4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
                -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
                7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
                2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
                1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
                -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
                -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
                -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
                10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
                -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
                -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
                7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
                -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
                -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
                -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
                12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
                -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
                -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
                11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
                4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
                5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
                3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
                -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
                -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
                -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
                -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
                -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
                -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
                5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
                5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
                1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
                9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
                4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
                2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
                -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
                -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
                4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
                0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
                -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
                -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
                -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
                8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
                0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
                7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
                -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
                10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
                -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
                10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
                -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
                -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
                3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
                5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
                -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
                3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
                2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
                -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
                -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
                -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
                -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
                6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
                -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
                -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
                -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
                3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
                -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
                -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
                2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
                -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
                -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
                5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
                -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
                -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
                -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
                10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
                7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
                -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
                -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
                7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
                -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
                -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
                -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
                7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
                -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
                1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
                2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
                -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
                -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
                7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
                1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
                9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
                -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
                -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
                7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
                12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
                6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
                5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
                2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
                3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
                2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
                9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
                -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
                -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
                1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
                6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
                2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
                6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
                3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
                7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
                -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
                -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
                -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
                -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
                8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
                4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
                -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
                4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
                -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
                -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
                7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
                -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
                -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
                8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
                -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
                1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
                7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
                -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
                11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
                -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
                3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
                5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
                0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
                -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
                0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
                -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
                5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
                3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
                -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
                -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
                -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
                6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
                -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
                -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
                1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
                4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
                -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
                2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
                -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
                4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
                -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
                -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
                7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
                4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
                -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
                7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
                7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
                -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
                -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
                -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
                2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
                10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
                -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
                8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
                2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
                -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
                -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
                -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
                5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
                -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
                -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
                -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
                -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
                -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
                2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
                -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
                -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
                -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
                -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
                6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
                -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
                11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
                7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
                -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
                -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
                -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
                -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
                -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
                -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
                -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
                -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
                1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
                1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
                9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
                5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
                -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
                -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
                -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
                -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
                8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
                2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
                7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
                -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
                -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
                4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
                3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
                -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
                5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
                4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
                -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
                0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
                -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
                3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
                -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
                8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
                -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
                2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
                10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
                6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
                -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
                -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
                -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
                -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
                -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
                4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
                2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
                6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
                3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
                11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
                -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
                4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
                2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
                -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
                -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
                -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
                6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
                0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
                -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
                -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
                -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
                5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
                2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
                -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
                9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
                11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
                3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
                -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
                3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
                -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
                5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
                8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
                7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
                -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
                7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
                9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
                7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
                -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
        };

// Constructor of ORBextractor
ORBextractor::ORBextractor(int _nfeatures,      // desired number of features
                           float _scaleFactor,  // scale factor between pyramid levels
                           int _nlevels,        // number of pyramid levels
                           int _iniThFAST,      // initial FAST threshold (strong corners)
                           int _minThFAST):     // lower FAST threshold (for low-texture regions)
    nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
    iniThFAST(_iniThFAST), minThFAST(_minThFAST)
{
    // Allocate vector for per-level scale factor.
    mvScaleFactor.resize(nlevels);
    // Store sigma^2 for each level (actually the square of the relative scale).
    mvLevelSigma2.resize(nlevels);
    // Level 0 has scale factor and sigma^2 equal to 1.
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    // Compute scale factors for the rest of the pyramid.
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    // Precompute inverse scale factors and inverse sigma^2.
    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    // Allocate pyramid images.
    mvImagePyramid.resize(nlevels);

    // Allocate per-level feature count.
    mnFeaturesPerLevel.resize(nlevels);

    // Inverse scale factor for distributing features per scale.
    float factor = 1.0f / scaleFactor;
    // Desired number of features per scale unit.
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    // Accumulate the assigned features per level.
    int sumFeatures = 0;
    // Assign features to each level except the last (top) level.
    for( int level = 0; level < nlevels-1; level++ )
    {
        // Round to nearest integer.
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    // Remaining features are assigned to the top level.
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    // Number of points in the BRIEF pattern; 512 = 256 pairs of points.
    const int npoints = 512;
    // Pattern pointer. bit_pattern_31_ is int[], we reinterpret it as Point*.
    const Point* pattern0 = (const Point*)bit_pattern_31_;
    // Copy global int pattern into the class' pattern vector (cv::Point).
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

    // This is for orientation computation
    // Pre-compute the row end for each v in a circular patch.
    // "+1" here includes the center row.
    umax.resize(HALF_PATCH_SIZE + 1);

    // cvFloor: largest integer <= x, cvCeil: smallest integer >= x, cvRound: round to nearest.
    int v,
        v0,
        vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1); // max row index to explicitly compute
                         // This corresponds to points on a 45° arc of the circle.

    // Compute minimum v to start mirroring using symmetry.
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    // Squared radius.
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;

    // Compute umax[v] = horizontal radius at each row v (x coordinate bound).
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v)); // always non-negative

    // Make sure the pattern is perfectly symmetric.
    // We reuse symmetry of the circle so that rounding by cvRound
    // does not break rotation invariance of the sampling pattern.
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}


/**
 * @brief Compute the orientation of all keypoints on one image.
 *
 * @param[in] image                 Image at the current pyramid level.
 * @param[in,out] keypoints         Keypoints whose angles will be updated.
 * @param[in] umax                  Precomputed row bounds for the circular patch.
 */
static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax)
{
    // Iterate over all keypoints.
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        // Compute orientation using intensity centroid.
        keypoint->angle = IC_Angle(image,
                                   keypoint->pt,
                                   umax);
    }
}


/**
 * @brief Divide the current ExtractorNode into four child nodes, and
 *        distribute keypoints into the corresponding subregions.
 *
 * @param[in,out] n1  Child node 1: upper-left
 * @param[in,out] n2  Child node 2: upper-right
 * @param[in,out] n3  Child node 3: bottom-left
 * @param[in,out] n4  Child node 4: bottom-right
 */
void ExtractorNode::DivideNode(ExtractorNode &n1,
                               ExtractorNode &n2,
                               ExtractorNode &n3,
                               ExtractorNode &n4)
{
    // Half width / height of this node's image region.
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    // Define boundaries of child nodes.
    // n1: upper-left region
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    // Reserve storage for keypoints in this child.
    n1.vKeys.reserve(vKeys.size());

    // n2: upper-right region
    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    // n3: bottom-left region
    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    // n4: bottom-right region
    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    // Associate existing keypoints to child nodes according to their location.
    for(size_t i=0;i<vKeys.size();i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        // Compare with child region bounds and push into the corresponding node.
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    // Mark child nodes that contain only one keypoint as non-divisible.
    if(n1.vKeys.size()==1)
        n1.bNoMore = true;
    if(n2.vKeys.size()==1)
        n2.bNoMore = true;
    if(n3.vKeys.size()==1)
        n3.bNoMore = true;
    if(n4.vKeys.size()==1)
        n4.bNoMore = true;
}

static bool compareNodes(pair<int,ExtractorNode*>& e1, pair<int,ExtractorNode*>& e2)
{
    if(e1.first < e2.first)
    {
        return true;
    }
    else if(e1.first > e2.first)
    {
        return false;
    }
    else
    {
        if(e1.second->UL.x < e2.second->UL.x)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

/**
 * @brief Uniformly distribute keypoints on one pyramid level using a quadtree.
 *
 * @param[in] vToDistributeKeys     Keypoints to be distributed.
 * @param[in] minX                  Left boundary of this level's valid image region
 *                                  (in "expanded image" coordinates).
 * @param[in] maxX                  Right boundary.
 * @param[in] minY                  Top boundary.
 * @param[in] maxY                  Bottom boundary.
 * @param[in] N                     Desired number of keypoints on this level.
 * @param[in] level                 Pyramid level index (not used).
 * @return vector<cv::KeyPoint>     Keypoints after spatial redistribution.
 */
vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                       const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
    // Compute how many initial nodes along x direction.
    // Step 1: determine number of initial nodes based on aspect ratio.
    // WARNING: if (maxX-minX)/(maxY-minY) < 0.5, nIni may become 0 and later cause division by zero.
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

    // Horizontal size of each initial node.
    const float hX = static_cast<float>(maxX-minX)/nIni;

    // List of all active nodes.
    list<ExtractorNode> lNodes;

    // Pointers to initial nodes.
    vector<ExtractorNode*> vpIniNodes;

    vpIniNodes.resize(nIni);

    // Step 2: create initial nodes.
    for(int i=0; i<nIni; i++)
    {
        ExtractorNode ni;

        // Node boundaries (in "expanded image" coordinates).
        // The keypoints are in the "expanded image" coordinate system.
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);    // UpLeft
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);  // UpRight
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY);             // BottomLeft
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);             // BottomRight

        // Reserve storage space.
        ni.vKeys.reserve(vToDistributeKeys.size());

        // Push node (copy of ni) to the list.
        lNodes.push_back(ni);
        // Store a pointer to the node just inserted.
        vpIniNodes[i] = &lNodes.back();
    }

    // Associate keypoints to initial nodes according to x position.
    // Step 3: assign each keypoint to a node.
    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }

    // Step 4: mark nodes that cannot be split and remove empty ones.
    // Note: this extra step could be done via direct counting, but it makes
    // the later logic clearer.
    list<ExtractorNode>::iterator lit = lNodes.begin();
    while(lit!=lNodes.end())
    {
        // Node with a single keypoint: cannot subdivide further.
        if(lit->vKeys.size()==1)
        {
            lit->bNoMore=true;
            lit++;
        }
        // Node with no keypoints: remove it.
        else if(lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    // Termination flag.
    bool bFinish = false;

    // Iteration counter (for debugging / statistics only).
    int iteration = 0;

    // For each iteration, store how many keypoints are in each node that can still be split.
    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;

    // Reserve some space for the worst case (each node split into four children).
    vSizeAndPointerToNode.reserve(lNodes.size()*4);

    // Step 5: recursively subdivide nodes using a quadtree until
    // the desired number of keypoints is reached.
    while(!bFinish)
    {
        iteration++;

        // Number of nodes before subdivision.
        int prevSize = lNodes.size();

        lit = lNodes.begin();

        // Number of nodes that can be expanded (have more than one keypoint).
        int nToExpand = 0;

        // Clear data from the previous iteration.
        vSizeAndPointerToNode.clear();

        // Subdivide all expandable nodes.
        while(lit!=lNodes.end())
        {
            // If a node cannot be split (one keypoint), skip it.
            if(lit->bNoMore)
            {
                lit++;
                continue;
            }
            else
            {
                // Node with more than one keypoint: subdivide it.
                ExtractorNode n1,n2,n3,n4;

                lit->DivideNode(n1,n2,n3,n4);

                // Add children that contain keypoints.
                if(n1.vKeys.size()>0)
                {
                    lNodes.push_front(n1);

                    if(n1.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                // Remove the parent node after expansion.
                // Note: this is effectively a first-level child node after the root.
                lit=lNodes.erase(lit);

                continue;
            }
        }

        // Finish if there are enough nodes or further subdivision is not possible.
        if((int)lNodes.size()>=N
            || (int)lNodes.size()==prevSize) // no node was split -> all nodes have size 1
        {
            bFinish = true;
        }

        // Step 6: If subdividing all expandable nodes once more would exceed N,
        // perform a controlled subdivision to get close to N.
        // There is some subtle complexity here due to accumulated nToExpand, but
        // the main idea is to stop as soon as we reach or slightly exceed N.
        if(!bFinish && ((int)lNodes.size()+nToExpand*3)>N)
        {
            // Refine subdivision so that final number of nodes is just >= N.
            while(!bFinish)
            {
                prevSize = lNodes.size();

                // Make a copy of nodes that were expandable in this iteration.
                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                // Sort nodes to split (largest first) so that dense areas are split first.
                // Note: this order affects which final keypoints survive.
                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end(),compareNodes);
                // stable_sort(...) could also be used if deterministic behavior is desired.

                // Subdivide from the largest nodes backwards.
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    // Add children that contain keypoints.
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    // Remove parent node.
                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    // Stop if we have reached or exceeded N.
                    if((int)lNodes.size()>=N)
                        break;
                }

                // If after this pass we still didn't reach N or no node was split,
                // we are done.
                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;
            }
        }
    }

    // Step 7: For each node, keep only the keypoint with the highest response.
    vector<cv::KeyPoint> vResultKeys;

    vResultKeys.reserve(nfeatures);

    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;

        cv::KeyPoint* pKP = &vNodeKeys[0];

        float maxResponse = pKP->response;

        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    // vResultKeys now contains one keypoint per node, with uniform spatial distribution.
    return vResultKeys;
}


// Compute keypoints using the quadtree-based approach.
void ORBextractor::ComputeKeyPointsOctTree(
    vector<vector<KeyPoint> >& allKeypoints) // all keypoints across pyramid levels
{
    // Resize outer vector to number of pyramid levels.
    allKeypoints.resize(nlevels);

    // Size of each grid cell (square) in pixels.
    const float W = 35;

    // Process every pyramid level.
    for (int level = 0; level < nlevels; ++level)
    {
        // Compute valid keypoint coordinate range for this level.
        // EDGE_THRESHOLD is the margin used for descriptor computation and border replication.
        const int minBorderX = EDGE_THRESHOLD-3;  // minus 3 for FAST circle radius
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

        // Keypoints detected in all grid cells at this scale, before quadtree distribution.
        vector<cv::KeyPoint> vToDistributeKeys;
        // Over-sample: reserve ~10x of target features per image for safety.
        vToDistributeKeys.reserve(nfeatures*10);

        // Extent of valid region.
        const float width = (maxBorderX-minBorderX);
        const float height = (maxBorderY-minBorderY);

        // Number of cells horizontally and vertically.
        const int nCols = width/W;
        const int nRows = height/W;
        // Pixel size of each cell (ceil to cover the region).
        const int wCell = ceil(width/nCols);
        const int hCell = ceil(height/nRows);

        // Iterate over the grid of cells.
        for(int i=0; i<nRows; i++)
        {
            // Row start in image coordinates.
            const float iniY =minBorderY+i*hCell;
            // Row end (add 6 = 3+3, to allow for FAST circle radius at borders).
            float maxY = iniY+hCell+6;

            // If the start is already outside valid image border (with FAST radius), skip.
            if(iniY>=maxBorderY-3)
                continue;
            // Clamp to border.
            if(maxY>maxBorderY)
                maxY = maxBorderY;

            for(int j=0; j<nCols; j++)
            {
                // Column start.
                const float iniX =minBorderX+j*wCell;
                // Column end (+6 for FAST radius).
                float maxX = iniX+wCell+6;
                // If start is beyond valid area, skip.
                // NOTE: There is a potential minor bug here; ideally should compare with (maxBorderX-3).
                if(iniX>=maxBorderX-3)
                    continue;
                // Clamp to border.
                if(maxX>maxBorderX)
                    maxX = maxBorderX;

                // FAST keypoint extraction with high threshold.
                vector<cv::KeyPoint> vKeysCell;
                FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                     vKeysCell,
                     iniThFAST,
                     true);

                // If no keypoints with high threshold, try lower threshold.
                if(vKeysCell.empty())
                {
                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                         vKeysCell,
                         minThFAST,
                         true);
                }

                // If we got some keypoints in this cell, shift them to image coordinates.
                if(!vKeysCell.empty())
                {
                    for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                    {
                        // Currently vit->pt is relative to the cell block; shift it
                        // back to coordinates of the expanded image at this level.
                        (*vit).pt.x+=j*wCell;
                        (*vit).pt.y+=i*hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }
            }
        }

        // Reference to keypoints of this level in the output structure.
        vector<KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(nfeatures);

        // Apply quadtree based spatial distribution.
        keypoints = DistributeOctTree(vToDistributeKeys,
                                      minBorderX, maxBorderX,
                                      minBorderY, maxBorderY,
                                      mnFeaturesPerLevel[level],
                                      level);

        // Compute patch size at this level (for orientation and descriptor).
        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Add the border offset and set octave + size fields.
        const int nkps = keypoints.size();
        for(int i=0; i<nkps ; i++)
        {
            // Shift to coordinates of the expanded pyramid image at this level.
            keypoints[i].pt.x+=minBorderX;
            keypoints[i].pt.y+=minBorderY;
            // Pyramid level index.
            keypoints[i].octave=level;
            // Patch size (a.k.a. keypoint radius) in pixels at this scale.
            keypoints[i].size = scaledPatchSize;
        }
    }

    // Compute orientations for all keypoints on all levels.
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level],
                           allKeypoints[level],
                           umax);
}


// Compute keypoints using the original grid-based method (unused in default code).
void ORBextractor::ComputeKeyPointsOld(
    std::vector<std::vector<KeyPoint> > &allKeypoints)
{
    allKeypoints.resize(nlevels);

    // Image aspect ratio at level 0 (same for all levels).
    float imageRatio = (float)mvImagePyramid[0].cols/mvImagePyramid[0].rows;

    for (int level = 0; level < nlevels; ++level)
    {
        // Desired number of features at this level.
        const int nDesiredFeatures = mnFeaturesPerLevel[level];

        // Compute number of grid cells along x and y.
        // Note: implicit float->int conversion truncates, equivalent to floor.
        const int levelCols = sqrt((float)nDesiredFeatures/(5*imageRatio));
        const int levelRows = imageRatio*levelCols;

        // Valid coordinate range for FAST keypoints (exclude border).
        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD;

        // Width and height of the valid region.
        const int W = maxBorderX - minBorderX;
        const int H = maxBorderY - minBorderY;
        // Cell size.
        const int cellW = ceil((float)W/levelCols);
        const int cellH = ceil((float)H/levelRows);

        // Total number of cells.
        const int nCells = levelRows*levelCols;
        // Desired number of features per cell (initial).
        const int nfeaturesCell = ceil((float)nDesiredFeatures/nCells);

        // 3D vector: [row][col] -> vector of keypoints in that cell.
        vector<vector<vector<KeyPoint> > > cellKeyPoints(levelRows, vector<vector<KeyPoint> >(levelCols));

        // Number of keypoints to retain in each cell.
        vector<vector<int> > nToRetain(levelRows,vector<int>(levelCols,0));
        // Total number of keypoints detected in each cell.
        vector<vector<int> > nTotal(levelRows,vector<int>(levelCols,0));
        // Whether the cell cannot provide more keypoints (true if total <= requested).
        vector<vector<bool> > bNoMore(levelRows,vector<bool>(levelCols,false));
        // Top-left coordinates of each cell (in expanded image coordinates).
        vector<int> iniXCol(levelCols);
        vector<int> iniYRow(levelRows);

        // Number of cells that ran out of keypoints (cannot contribute more).
        int nNoMore = 0;
        // Remaining number of keypoints to be distributed.
        int nToDistribute = 0;

        // Height of the region used for FAST (cellH + 6: ±3 for FAST circle).
        float hY = cellH + 6;

        // Iterate row by row.
        for(int i=0; i<levelRows; i++)
        {
            // Top coordinate for this row of cells (minus 3 for FAST radius).
            const float iniY = minBorderY + i*cellH - 3;
            iniYRow[i] = iniY;

            // For the last row, recompute height precisely up to maxBorderY.
            if(i == levelRows-1)
            {
                hY = maxBorderY+3-iniY;
                // If negative, this row does not exist in valid region.
                if(hY<=0)
                    continue;
            }

            float hX = cellW + 6;

            for(int j=0; j<levelCols; j++)
            {
                float iniX;

                if(i==0)
                {
                    // First row: compute iniX and store it for later rows.
                    iniX = minBorderX + j*cellW - 3;
                    iniXCol[j] = iniX;
                }
                else
                {
                    // Other rows: reuse iniX.
                    iniX = iniXCol[j];
                }

                if(j == levelCols-1)
                {
                    hX = maxBorderX+3-iniX;
                    if(hX<=0)
                        continue;
                }

                // Extract this cell region.
                Mat cellImage = mvImagePyramid[level].rowRange(iniY,iniY+hY).colRange(iniX,iniX+hX);

                // Over-sample a bit inside the cell to ensure enough keypoints.
                cellKeyPoints[i][j].reserve(nfeaturesCell*5);

                // First FAST pass with higher threshold.
                FAST(cellImage,
                     cellKeyPoints[i][j],
                     iniThFAST,
                     true);

                // If very few keypoints, try lower threshold.
                if(cellKeyPoints[i][j].size()<=3)
                {
                    cellKeyPoints[i][j].clear();
                    FAST(cellImage,
                         cellKeyPoints[i][j],
                         minThFAST,
                         true);
                }

                const int nKeys = cellKeyPoints[i][j].size();
                nTotal[i][j] = nKeys;

                if(nKeys>nfeaturesCell)
                {
                    // Enough keypoints: we will retain nfeaturesCell of them.
                    nToRetain[i][j] = nfeaturesCell;
                    bNoMore[i][j] = false;
                }
                else
                {
                    // Not enough: we retain all, and record the deficit to be
                    // redistributed to other cells.
                    nToRetain[i][j] = nKeys;
                    nToDistribute += nfeaturesCell-nKeys;
                    bNoMore[i][j] = true;
                    nNoMore++;
                }
            }
        }


        // Retain by response score, iteratively balancing cells.
        // This loop tries to redistribute the deficit nToDistribute across cells that
        // have surplus keypoints until there is no more deficit or all cells are exhausted.
        while(nToDistribute>0 && nNoMore<nCells)
        {
            // New target per cell for this iteration.
            int nNewFeaturesCell = nfeaturesCell + ceil((float)nToDistribute/
                                                        (nCells-nNoMore));

            // Reset deficit; it will be recomputed as we check each cell.
            nToDistribute = 0;

            for(int i=0; i<levelRows; i++)
            {
                for(int j=0; j<levelCols; j++)
                {
                    // Only cells that had enough keypoints in previous round are considered.
                    if(!bNoMore[i][j])
                    {
                        if(nTotal[i][j]>nNewFeaturesCell)
                        {
                            // Enough to meet the new target.
                            nToRetain[i][j] = nNewFeaturesCell;
                            bNoMore[i][j] = false;
                        }
                        else
                        {
                            // Still not enough, keep all we have and mark deficit again.
                            nToRetain[i][j] = nTotal[i][j];
                            nToDistribute += nNewFeaturesCell-nTotal[i][j];
                            bNoMore[i][j] = true;
                            nNoMore++;
                        }
                    }
                }
            }
        }

        // Now we have decided for each cell how many keypoints should be retained.
        // Next, we actually filter the keypoints.

        vector<KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(nDesiredFeatures*2);

        // Compute scaled patch size for this level.
        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Retain by score and transform coordinates.
        for(int i=0; i<levelRows; i++)
        {
            for(int j=0; j<levelCols; j++)
            {
                vector<KeyPoint> &keysCell = cellKeyPoints[i][j];
                // Keep the best nToRetain[i][j] keypoints in this cell (highest response).
                KeyPointsFilter::retainBest(keysCell,
                                            nToRetain[i][j]);
                // Extra safety: if still more than requested, cut the tail.
                if((int)keysCell.size()>nToRetain[i][j])
                    keysCell.resize(nToRetain[i][j]);

                // Transform cell-local coordinates to expanded image coordinates
                // and store octave/size information.
                for(size_t k=0, kend=keysCell.size(); k<kend; k++)
                {
                    keysCell[k].pt.x+=iniXCol[j];
                    keysCell[k].pt.y+=iniYRow[i];
                    keysCell[k].octave=level;
                    keysCell[k].size = scaledPatchSize;
                    keypoints.push_back(keysCell[k]);
                }
            }
        }

        // If we still have more than desired features overall, keep the best globally.
        if((int)keypoints.size()>nDesiredFeatures)
        {
            KeyPointsFilter::retainBest(keypoints,nDesiredFeatures);
            keypoints.resize(nDesiredFeatures);
        }
    }

    // Finally, compute orientations at all levels.
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level],
                           allKeypoints[level],
                           umax);
}

// This is a file-local static function, not a member of any class.
/**
 * @brief Compute descriptors for all keypoints on one pyramid level.
 *
 * @param[in] image                 Image at one pyramid level.
 * @param[in] keypoints             Vector of keypoints.
 * @param[out] descriptors          Output matrix of descriptors (rows = keypoints).
 * @param[in] pattern               BRIEF sampling pattern.
 */
static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,
                               const vector<Point>& pattern)
{
    // Allocate descriptor matrix: num_keypoints x 32 bytes.
    descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

    // Compute descriptor for each keypoint.
    for (size_t i = 0; i < keypoints.size(); i++)
        computeOrbDescriptor(keypoints[i],
                             image,
                             &pattern[0],
                             descriptors.ptr((int)i));
}

/**
 * @brief Main call operator: compute ORB keypoints and descriptors.
 *
 * @param[in] _image                    Input image (grayscale).
 * @param[in] _mask                     Mask (unused here).
 * @param[in,out] _keypoints            Output vector of keypoints.
 * @param[in,out] _descriptors          Output descriptors matrix.
 * @param[in,out] vLappingArea          Region [x_min, x_max] for stereo overlap (used
 *                                      to separate mono and stereo keypoints).
 * @return int                          Number of keypoints assigned to the mono part.
 */
int ORBextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                              OutputArray _descriptors, std::vector<int> &vLappingArea)
{
    // Step 1: Validate input.
    if(_image.empty())
        return -1;

    Mat image = _image.getMat();
    // Expect single-channel 8-bit image.
    // assert(image.type() == CV_8UC1 );

    // Step 2: Build image pyramid.
    ComputePyramid(image);

    // Step 3: Detect and spatially distribute keypoints on each pyramid level.
    vector < vector<KeyPoint> > allKeypoints;
    // Quadtree-based distribution.
    ComputeKeyPointsOctTree(allKeypoints);

    // Old method (unused by default).
    // ComputeKeyPointsOld(allKeypoints);


    // Step 4: Allocate output descriptor matrix.
    Mat descriptors;

    // Count total keypoints across all levels.
    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();

    if( nkeypoints == 0 )
        _descriptors.release();
    else
    {
        // Create matrix for all descriptors (all levels).
        _descriptors.create(nkeypoints,
                            32,
                            CV_8U);
        // We will fill this matrix row by row.
        descriptors = _descriptors.getMat();
    }

    _keypoints = vector<cv::KeyPoint>(nkeypoints);

    // Offset for writing into the descriptor matrix.
    int offset = 0;
    // Indices for mono and stereo parts.
    int monoIndex = 0, stereoIndex = nkeypoints-1;
    for (int level = 0; level < nlevels; ++level)
    {
        vector<KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if(nkeypointsLevel==0)
            continue;

        // Step 5: Pre-blur the image at this level to reduce noise for descriptors.
        Mat workingMat = mvImagePyramid[level].clone();

        // For keypoint detection we used the original image; here we apply Gaussian
        // blur to stabilize descriptor computation.
        GaussianBlur(workingMat,
                     workingMat,
                     Size(7, 7),
                     2,
                     2,
                     BORDER_REFLECT_101);

        // Compute descriptors for this level.
        Mat desc = cv::Mat(nkeypointsLevel, 32, CV_8U);
        computeDescriptors(workingMat,
                           keypoints,
                           desc,
                           pattern);

        offset += nkeypointsLevel;

        // Step 6: Scale keypoint coordinates back to level 0 for all levels.
        float scale = mvScaleFactor[level];
        int i = 0;
        for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                     keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint){

            if (level != 0){
                // Rescale keypoint coordinates to original image.
                keypoint->pt *= scale;
            }
            // vLappingArea is a stereo overlap region [x_min, x_max]. Keypoints
            // inside it are used for stereo; outside for mono.
            if(keypoint->pt.x >= vLappingArea[0] && keypoint->pt.x <= vLappingArea[1]){
                _keypoints.at(stereoIndex) = (*keypoint);
                desc.row(i).copyTo(descriptors.row(stereoIndex));
                stereoIndex--;
            }
            else{
                _keypoints.at(monoIndex) = (*keypoint);
                desc.row(i).copyTo(descriptors.row(monoIndex));
                monoIndex++;
            }
            i++;
        }
    }
    // cout << "[ORBextractor]: extracted " << _keypoints.size() << " KeyPoints" << endl;
    return monoIndex;
}

/**
 * @brief Build the image pyramid.
 *
 * @param image Input base image. All pixels are valid for FAST detection (before padding).
 */
void ORBextractor::ComputePyramid(cv::Mat image)
{
    for (int level = 0; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level];
        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));

        Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        // Alternative formula (commented):
        // Size wholeSize(cvRound(((float)image.cols+EDGE_THRESHOLD)*scale),
        //                cvRound(((float)image.rows+EDGE_THRESHOLD)*scale));

        Mat temp(wholeSize, image.type()), masktemp;
        // The ROI inside temp where the actual (resized) image resides.
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image for this level.
        if( level != 0 )
        {
            // Resize previous level to current size.
            resize(mvImagePyramid[level-1],
                   mvImagePyramid[level],
                   sz,
                   0,
                   0,
                   cv::INTER_LINEAR);

            // Copy into temp with a border of EDGE_THRESHOLD pixels around it.
            // This padding is used to safely compute descriptors near the borders.
            copyMakeBorder(mvImagePyramid[level],
                           temp,
                           EDGE_THRESHOLD, EDGE_THRESHOLD,
                           EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101+BORDER_ISOLATED);

            /*
             * Border types (OpenCV docs):
             * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
             * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
             * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
             * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
             * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  (some constant 'i')
             *
             * BORDER_ISOLATED indicates that the border extrapolation is done
             * only within this ROI.
             */
        }
        else
        {
            // Level 0: copy original image and expand borders.
            copyMakeBorder(image,
                           temp,
                           EDGE_THRESHOLD, EDGE_THRESHOLD,
                           EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101);
        }
    }
}

} //namespace ORB_SLAM3
