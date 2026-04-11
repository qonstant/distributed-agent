package telegram

import "regexp"

var commandRegexp = regexp.MustCompile(`^/(start|help|randompic)(@\w+)?`)
