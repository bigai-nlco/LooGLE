// Dean Attali / Beautiful Jekyll 2023



let BeautifulJekyllJS = {

    bigImgEl: null,
    numImgs: null,

    init: function () {
        setTimeout(BeautifulJekyllJS.initNavbar, 10);

        // Shorten the navbar after scrolling a little bit down
        $(window).scroll(function () {
            if ($(".navbar").offset().top > 50) {
                $(".navbar").addClass("top-nav-short");
            } else {
                $(".navbar").removeClass("top-nav-short");
            }
        });

        // On mobile, hide the avatar when expanding the navbar menu
        $('#main-navbar').on('show.bs.collapse', function () {
            $(".navbar").addClass("top-nav-expanded");
        });
        $('#main-navbar').on('hidden.bs.collapse', function () {
            $(".navbar").removeClass("top-nav-expanded");
        });

        // show the big header image
        BeautifulJekyllJS.initImgs();

        BeautifulJekyllJS.initPres();

        BeautifulJekyllJS.initSearch();

        document.querySelectorAll("table").forEach(function (table) {
            let wrapper = document.createElement("div");
            wrapper.className = "table-responsive";
            table.parentNode.insertBefore(wrapper, table);
            table.parentNode.replaceChild(wrapper, table);
            wrapper.appendChild(table);
            table.className = "tg table table-bordered table-striped table-hover";
            // table.querySelectorAll("th").forEach(function (td) {
            //     td.removeAttribute("class");
            // });

            // table.querySelectorAll("td").forEach(function (td) {
            //     td.removeAttribute("class");
            // });
        });


    },

    initNavbar: function () {
        // Set the navbar-dark/light class based on its background color
        const rgb = $('.navbar').css("background-color").replace(/[^\d,]/g, '').split(",");
        const brightness = Math.round(( // http://www.w3.org/TR/AERT#color-contrast
            parseInt(rgb[0]) * 299 +
            parseInt(rgb[1]) * 587 +
            parseInt(rgb[2]) * 114
        ) / 1000);
        if (brightness <= 125) {
            $(".navbar").removeClass("navbar-light").addClass("navbar-dark");
        } else {
            $(".navbar").removeClass("navbar-dark").addClass("navbar-light");
        }
    },

    initImgs: function () {
        // If the page was large images to randomly select from, choose an image
        if ($("#header-big-imgs").length > 0) {
            BeautifulJekyllJS.bigImgEl = $("#header-big-imgs");
            BeautifulJekyllJS.numImgs = BeautifulJekyllJS.bigImgEl.attr("data-num-img");

            // 2fc73a3a967e97599c9763d05e564189
            // set an initial image
            const imgInfo = BeautifulJekyllJS.getImgInfo();
            const src = imgInfo.src;
            const desc = imgInfo.desc;
            BeautifulJekyllJS.setImg(src, desc);

            // For better UX, prefetch the next image so that it will already be loaded when we want to show it
            const getNextImg = function () {
                const imgInfo = BeautifulJekyllJS.getImgInfo();
                const src = imgInfo.src;
                const desc = imgInfo.desc;

                const prefetchImg = new Image();
                prefetchImg.src = src;
                // if I want to do something once the image is ready: `prefetchImg.onload = function(){}`

                setTimeout(function () {
                    const img = $("<div></div>").addClass("big-img-transition").css("background-image", 'url(' + src + ')');
                    $(".intro-header.big-img").prepend(img);
                    setTimeout(function () { img.css("opacity", "1"); }, 50);

                    // after the animation of fading in the new image is done, prefetch the next one
                    //img.one("transitioned webkitTransitionEnd oTransitionEnd MSTransitionEnd", function(){
                    setTimeout(function () {
                        BeautifulJekyllJS.setImg(src, desc);
                        img.remove();
                        getNextImg();
                    }, 1000);
                    //});
                }, 6000);
            };

            // If there are multiple images, cycle through them
            if (BeautifulJekyllJS.numImgs > 1) {
                getNextImg();
            }
        }
    },

    initPres: function () {
        const copyButton = "copy";
        let blocks = document.querySelectorAll("pre");
        blocks.forEach(function (block) {
            if (navigator.clipboard) {
                let button_container = document.createElement("div");
                button_container.className = "clipboard-btn-container";
                let button = document.createElement("button");
                button.className = "btn clipboard-btn";
                button.innerHTML =
                    '<svg aria-hidden="true" height="16" viewBox="0 0 16 16" version ="1.1" width ="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon m-2" > <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path>< path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z" ></path > <path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path></svg >';
                // button.innerText = "copy";
                button_container.appendChild(button);
                block.appendChild(button_container);
                button.addEventListener("click", async function () {
                    await copyCode(block);
                });
            }
        });

    },

    getImgInfo: function () {
        const randNum = Math.floor((Math.random() * BeautifulJekyllJS.numImgs) + 1);
        const src = BeautifulJekyllJS.bigImgEl.attr("data-img-src-" + randNum);
        const desc = BeautifulJekyllJS.bigImgEl.attr("data-img-desc-" + randNum);

        return {
            src: src,
            desc: desc
        }
    },

    setImg: function (src, desc) {
        $(".intro-header.big-img").css("background-image", 'url(' + src + ')');
        if (typeof desc !== typeof undefined && desc !== false) {
            $(".img-desc").text(desc).show();
        } else {
            $(".img-desc").hide();
        }
    },

    initSearch: function () {
        if (!document.getElementById("beautifuljekyll-search-overlay")) {
            return;
        }

        $("#nav-search-link").click(function (e) {
            e.preventDefault();
            $("#beautifuljekyll-search-overlay").show();
            $("#nav-search-input").focus().select();
            $("body").addClass("overflow-hidden");
        });
        $("#nav-search-exit").click(function (e) {
            e.preventDefault();
            $("#beautifuljekyll-search-overlay").hide();
            $("body").removeClass("overflow-hidden");
        });
        $(document).on('keyup', function (e) {
            if (e.key == "Escape") {
                $("#beautifuljekyll-search-overlay").hide();
                $("body").removeClass("overflow-hidden");
            }
        });
    }
};

async function copyCode (block) {
    const code = block.querySelector("code");
    let text = code.innerText;
    await navigator.clipboard.writeText(text);
};

// 2fc73a3a967e97599c9763d05e564189

document.addEventListener('DOMContentLoaded', BeautifulJekyllJS.init);