// Downloads the datasets used by the course into data/.
//
// Usage, from the project root:
//
//	go run ./download          # everything
//	go run ./download mnist    # only one dataset
//
// Files that already exist are skipped. Each lesson adds its dataset here.
package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

type dataset struct {
	name  string   // folder inside data/
	base  string   // URL the files are downloaded from
	files []string // files to download
}

var datasets = []dataset{
	{
		name: "mnist", // handwritten digits (lessons 0, 5 and 7-8)
		base: "https://storage.googleapis.com/cvdf-datasets/mnist/",
		files: []string{
			"train-images-idx3-ubyte.gz",
			"train-labels-idx1-ubyte.gz",
			"t10k-images-idx3-ubyte.gz",
			"t10k-labels-idx1-ubyte.gz",
		},
	},
}

func main() {
	wanted := os.Args[1:]
	found := false

	for _, d := range datasets {
		if len(wanted) > 0 && !contains(wanted, d.name) {
			continue
		}
		found = true

		dir := filepath.Join("data", d.name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			fail(err)
		}
		for _, f := range d.files {
			path := filepath.Join(dir, f)
			if _, err := os.Stat(path); err == nil {
				fmt.Println("already there:", path)
				continue
			}
			fmt.Println("downloading:  ", path)
			if err := download(d.base+f, path); err != nil {
				fail(err)
			}
		}
	}

	if !found {
		fmt.Println("unknown dataset. Available:")
		for _, d := range datasets {
			fmt.Println("  " + d.name)
		}
		os.Exit(1)
	}
}

// download saves url into path. It writes to a temporary file first, so a
// failed download never leaves a half-written file behind.
func download(url, path string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: %s", url, resp.Status)
	}

	tmp := path + ".part"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, resp.Body); err != nil {
		out.Close()
		os.Remove(tmp)
		return err
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func fail(err error) {
	fmt.Println("error:", err)
	os.Exit(1)
}
