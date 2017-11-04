
__kernel void parallel_transforms(
                   __global const float *input,
                   __global float *output)
{

    /* --------------------------------------------------------------------- */
    uint tx = get_global_id(0);
    uint ty = get_global_id(1);
    /* Copy the transformation matrices to local memory. To prevent multiple
    accesses to global memory */

    //event_t event;
    //event = async_work_group_copy(transforms, transform, 8 * 4 * sizeof(int),0);
    //wait_group_events(1, &event);

    /* Hard coded transformations to save transfer time between local and global memory. 4 * 8 = 32. */
    const int transforms[32] = {1, 0, 0, 1, 0, 1, -1, 0, -1, 0, 0, -1, 0, -1, 1, 0, 0, 1, 1, 0, 0, -1, -1, 0, -1, 0, 0,
    1, 1, 0, 0, -1};
    /* --------------------------------------------------------------------- */

    float value = input[width * ty + tx];

    // Segment coordinates.
    float mod_tx = (tx % abst);
    float mod_ty = (ty % abst);

    // For shifting back.
    float global_shift_x = abs_diff(tx, mod_tx);
    float global_shift_y = abs_diff(ty, mod_ty);

    // Shift to centre of image.
    float co_tx = mod_tx - half_width;
    float co_ty = mod_ty - half_width;

    float new_tx;
    float new_ty;

    float tx_prime;
    float ty_prime;

    int index;

    // For each transform in transforms.
    for(int i = 0; i < 8; i++){
        // Transform and translate back to origin.
        new_tx = transforms[0 + 4 * i] * co_tx + transforms[1 + 4 * i] * co_ty + half_width;
        new_ty = transforms[2 + 4 * i] * co_tx + transforms[3 + 4 * i] * co_ty + half_width;

        // Then convert back to global co-ordinates.
        tx_prime = new_tx + global_shift_x;
        ty_prime = new_ty + global_shift_y;

        index = (int) width * ty_prime + tx_prime;
        output[index + i * width * width] = value;
    }

    /* --------------------------------------------------------------------- */

}