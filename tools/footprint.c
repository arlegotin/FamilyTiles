#include <errno.h>
#include <libproc.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: footprint PID\n");
        return 2;
    }
    char *end = NULL;
    errno = 0;
    long parsed = strtol(argv[1], &end, 10);
    if (errno != 0 || end == argv[1] || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
        fprintf(stderr, "invalid pid\n");
        return 2;
    }
    struct rusage_info_v4 info;
    memset(&info, 0, sizeof(info));
    if (proc_pid_rusage((int)parsed, RUSAGE_INFO_V4, (rusage_info_t *)&info) != 0) {
        fprintf(stderr, "proc_pid_rusage failed: %s\n", strerror(errno));
        return 1;
    }
    printf("%llu\n", (unsigned long long)info.ri_phys_footprint);
    return 0;
}
