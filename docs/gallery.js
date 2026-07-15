const dialog = document.querySelector("#image-lightbox");
const dialogImage = dialog.querySelector(".lightbox-image");
const dialogCaption = dialog.querySelector(".lightbox-caption");
const closeButton = dialog.querySelector(".lightbox-close");
const images = document.querySelectorAll(
  ".walkthrough-grid img, .gallery article > img",
);

let activeImage = null;

function openLightbox(image) {
  activeImage = image;
  dialogImage.src = image.currentSrc || image.src;
  dialogImage.alt = image.alt;
  dialogCaption.textContent = image.alt;
  document.body.classList.add("lightbox-open");
  dialog.showModal();
  closeButton.focus();
}

function closeLightbox() {
  dialog.close();
}

for (const image of images) {
  image.classList.add("lightbox-trigger");
  image.tabIndex = 0;
  image.setAttribute("role", "button");
  image.setAttribute("aria-haspopup", "dialog");
  image.setAttribute("aria-label", `Open full-size image: ${image.alt}`);
  image.addEventListener("click", () => openLightbox(image));
  image.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openLightbox(image);
    }
  });
}

closeButton.addEventListener("click", closeLightbox);
dialog.addEventListener("click", (event) => {
  if (event.target === dialog) {
    closeLightbox();
  }
});
dialog.addEventListener("close", () => {
  const trigger = activeImage;
  document.body.classList.remove("lightbox-open");
  dialogImage.removeAttribute("src");
  activeImage = null;
  window.requestAnimationFrame(() => trigger?.focus());
});
